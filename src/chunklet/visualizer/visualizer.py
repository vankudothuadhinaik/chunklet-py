import json
import mimetypes
import traceback
from pathlib import Path
from typing import Callable

import aiofiles
from pydantic import BaseModel, ConfigDict, Field, ValidationError

try:
    import msgpack
    import uvicorn
    from charset_normalizer import detect
    from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
    from fastapi.responses import HTMLResponse, Response
    from fastapi.staticfiles import StaticFiles
except ImportError:  # pragma: no cover
    # Lambda placeholders prevent "None is not callable" errors when imports fail
    # This allows the module to be imported without dependencies, with proper error handling later
    msgpack = None
    uvicorn = None
    detect = None
    FastAPI = None
    UploadFile = None
    File = lambda x: x  # noqa: E731
    Form = lambda x: x  # noqa: E731
    HTTPException = None
    Request = None
    HTMLResponse = lambda x: x  # noqa: E731
    Response = lambda x: x  # noqa: E731
    StaticFiles = None

from chunklet.code_chunker import CodeChunker
from chunklet.common.validation import validate_input
from chunklet.document_chunker import DocumentChunker


MAX_UPLOAD_BYTES = 10 * 1024 * 1024


class _UploadTooLarge(Exception):
    pass


class UploadSizeLimitMiddleware:
    """Reject oversized visualizer uploads before multipart parsing buffers them."""

    def __init__(self, app, max_bytes: int = MAX_UPLOAD_BYTES):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http" or scope.get("path") != "/api/chunk":
            await self.app(scope, receive, send)
            return

        content_length = next(
            (value for key, value in scope.get("headers", []) if key.lower() == b"content-length"),
            None,
        )
        if content_length is not None:
            try:
                if int(content_length) > self.max_bytes:
                    await self._send_too_large(scope, receive, send)
                    return
            except ValueError:
                pass

        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message.get("type") == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise _UploadTooLarge()
            return message

        try:
            await self.app(scope, limited_receive, send)
        except _UploadTooLarge:
            await self._send_too_large(scope, receive, send)

    async def _send_too_large(self, scope, receive, send):
        response = Response(
            content='{"detail":"Upload exceeds the 10 MiB request limit."}',
            status_code=413,
            media_type="application/json",
        )
        await response(scope, receive, send)


class _DocumentChunkParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lang: str | None = Field(default=None, min_length=1)
    max_tokens: int | None = Field(default=None, ge=12)
    max_sentences: int | None = Field(default=None, ge=1)
    max_section_breaks: int | None = Field(default=None, ge=1)
    overlap_percent: int | None = Field(default=None, ge=0, le=75)


class _CodeChunkParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_tokens: int | None = Field(default=None, ge=12)
    max_lines: int | None = Field(default=None, ge=5)
    max_functions: int | None = Field(default=None, ge=1)
    strict: bool = True

class Visualizer:
    """A FastAPI-based web interface for visualizing document and code chunks.

    This server allows users to upload text or code files, processes them with
    Chunklet's `DocumentChunker` or `CodeChunker`, and returns the chunked
    data along with statistics. A minimal frontend interface is served at the
    root endpoint.

    Attributes:
        host (str): Host IP to bind the FastAPI server.
        port (int): Port number to run the server on.
        document_chunker (DocumentChunker): Chunklet document chunker instance.
        code_chunker (CodeChunker): Chunklet code chunker instance.
        app (FastAPI): FastAPI application instance.
    """

    @validate_input
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        token_counter: Callable[[str], int] | None = None,
    ):
        """Initializes the Visualizer server and configures chunkers.

        Args:
            host: Host IP to run the server. Defaults to "127.0.0.1".
            port: Port number to run the server. Defaults to 8000.
            token_counter: Function to count tokens
                in text/code. Required for chunkers if used with `max_tokens`.
        """
        if FastAPI is None:
            raise ImportError(
                "The 'fastapi' library is not installed. "
                "Please install it with 'pip install fastapi>=0.115.12' or install the visualization extras "
                "with 'pip install 'chunklet-py[viz]''"
            )

        if msgpack is None:
            raise ImportError(
                "The 'msgpack' library is not installed. "
                "Please install it with 'pip install msgpack>=1.0.8' or install the visualization extras "
                "with 'pip install 'chunklet-py[viz]''"
            )

        self.host = host
        self.port = port
        self._token_counter = token_counter

        self.app = FastAPI()
        self.app.add_middleware(UploadSizeLimitMiddleware, max_bytes=MAX_UPLOAD_BYTES)

        # Initialize chunkers with sensible defaults; constraint attributes are
        # mutated per request from the submitted params.
        self.document_chunker = DocumentChunker(
            lang="en",
            token_counter=token_counter,
            max_sentences=3,
        )
        self.code_chunker = CodeChunker(token_counter=token_counter, max_lines=50)

        self.static_dir = Path(__file__).parent / "static"

        self.app.mount(
            "/static", StaticFiles(directory=str(self.static_dir)), name="static"
        )

        # API endpoints
        self.app.get("/api/token_counter_status")(self._get_token_counter_status)
        self.app.get("/health")(self._get_health_check)
        self.app.get("/")(self._get_index)
        self.app.post("/api/chunk")(self._chunk_file)

    # Instance endpoint methods
    def _get_token_counter_status(self):
        return {"token_counter_available": self.token_counter is not None}

    def _get_health_check(self):
        """Health check endpoint for testing."""
        return {"status": "healthy"}

    async def _get_index(self):
        """Serves the main HTML interface for the visualizer."""
        index_path = self.static_dir / "index.html"
        if index_path.exists():
            async with aiofiles.open(index_path, "r", encoding="utf-8") as f:
                content = await f.read()
                return HTMLResponse(content=content)
        return HTMLResponse(content="<h1>Text Chunk Visualizer</h1>")

    @validate_input
    async def _chunk_file(
        self,
        request: Request,
        file: UploadFile = File(...),
        mode: str = Form("document"),
        params: str = Form("{}"),
    ) -> Response:
        """Processes an uploaded file and returns chunked output.

        Args:
            self: The Visualizer instance.
            file: File uploaded by the client.
            mode: Determines which chunker to use ("document" or "code").
            params: JSON string containing chunking parameters.
            request: Optional Request object for content negotiation.

        Returns:
            MessagePack or JSON response with original text, chunks, and stats.
            Uses content negotiation: client can request MessagePack via
            Accept: application/msgpack header. Defaults to JSON for backward
            compatibility.

        Raises:
            HTTPException: If chunking fails.
        """
        # Validate the mode and accept only the documented per-mode parameters.
        if mode not in {"document", "code"}:
            raise HTTPException(400, "Mode must be either 'document' or 'code'.")

        try:
            raw_params = json.loads(params)
            if not isinstance(raw_params, dict):
                raise ValueError("Chunking parameters must be a JSON object.")
            params_model = (
                _DocumentChunkParams.model_validate(raw_params)
                if mode == "document"
                else _CodeChunkParams.model_validate(raw_params)
            )
            chunker_params = params_model.model_dump(
                exclude_unset=True,
                exclude_none=True,
            )
        except (json.JSONDecodeError, TypeError, ValueError, ValidationError):
            raise HTTPException(
                400,
                "Invalid chunking parameters for the selected mode.",
            ) from None
        # Use Python mimetypes instead of browser content_type for accuracy
        mimetype, _ = mimetypes.guess_type(file.filename or "")
        if not mimetype or not mimetype.startswith("text/"):
            raise HTTPException(400, "Only text files are supported.")

        if detect is None:
            raise HTTPException(
                400,
                "charset-normalizer library is not available. Please install visualization dependencies."
                "with 'pip install 'chunklet-py[viz]''",
            )

        content = await file.read()
        encoding = detect(content).get("encoding", "utf-8")
        text = content.decode(encoding, errors="ignore")

        try:
            if "max_tokens" in chunker_params and self._token_counter is None:
                raise HTTPException(
                    400,
                    "Token-based chunking requires a configured token counter.",
                )

            if mode == "code":
                # Build an isolated chunker per request so one browser session cannot
                # mutate the shared configuration used by another request.
                strict = chunker_params.pop("strict", True)
                code_options = {"max_lines": 50, **chunker_params}
                chunker = CodeChunker(
                    token_counter=self._token_counter,
                    **code_options,
                )
                chunks = [
                    dict(chunk) for chunk in chunker.chunk_text(text, strict=strict)
                ]
            else:
                document_options = {
                    "lang": "en",
                    "max_sentences": 3,
                    **chunker_params,
                }
                chunker = DocumentChunker(
                    token_counter=self._token_counter,
                    **document_options,
                )
                chunks = [dict(chunk) for chunk in chunker.chunk_text(text)]

            response_data = {
                "text": text,
                "chunks": chunks,
                "stats": {
                    "text_length": len(text),
                    "chunk_count": len(chunks),
                    "mode": mode,
                },
            }

            accept = request.headers.get("Accept", "") if request else ""
            if "application/msgpack" in accept:
                return Response(
                    content=msgpack.packb(response_data, use_bin_type=True),
                    media_type="application/msgpack",
                )

            return Response(
                content=json.dumps(response_data, ensure_ascii=False),
                media_type="application/json",
            )

        except HTTPException:
            raise
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(
                500,
                f"Chunking failed. Please check the server terminal for specific error details. ({str(e)})",
            ) from None

    @property
    def token_counter(self):
        """Get the current token counter function."""
        return self._token_counter

    @token_counter.setter
    @validate_input
    def token_counter(self, value):
        """Set the token counter and update both chunkers."""
        self._token_counter = value
        self.document_chunker.token_counter = value
        self.code_chunker.token_counter = value

    def serve(self):
        """Starts the FastAPI server and prints the server URL."""
        if uvicorn is None:
            raise ImportError(
                "The 'uvicorn' library is not installed. "
                "Please install it with 'pip install uvicorn>=0.34.0' or install the visualization extras "
                "with 'pip install 'chunklet-py[viz]''"
            )

        print(" =" * 20)
        print("\nTEXT CHUNK VISUALIZER")
        print("= " * 20)
        print(f"URL: http://{self.host}:{self.port}")

        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            access_log=False,
        )


if __name__ == "__main__":  # pragma: no cover
    visualizer = Visualizer()
    visualizer.serve()
