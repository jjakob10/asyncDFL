from __future__ import annotations

import logging
import threading
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import Response
from pydantic import BaseModel, Field

from app.config import Settings
from app.runtime import ClientRuntime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)


class StartRequest(BaseModel):
    run_id: str = Field(min_length=1, max_length=128)


def create_app(
    settings: Settings | None = None,
    runtime: ClientRuntime | None = None,
    auto_start_head: bool = True,
) -> FastAPI:
    configured_settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.runtime = runtime or ClientRuntime(configured_settings)

        if auto_start_head and configured_settings.is_head:
            def delayed_start() -> None:
                time.sleep(configured_settings.head_start_delay)
                app.state.runtime.start(str(uuid.uuid4()))

            threading.Thread(target=delayed_start, name="head-auto-start", daemon=True).start()
        yield

    app = FastAPI(
        title="Decentralized Federated Learning Client",
        version="1.0.0",
        lifespan=lifespan,
    )

    def get_runtime(request: Request) -> ClientRuntime:
        return request.app.state.runtime

    @app.get("/health")
    def health(request: Request) -> dict[str, object]:
        client = get_runtime(request)
        return {"ready": True, "client_id": client.settings.client_id}

    @app.post("/start", status_code=status.HTTP_202_ACCEPTED)
    def start(body: StartRequest, request: Request) -> dict[str, str]:
        result = get_runtime(request).start(body.run_id)
        if result == "conflict":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="this client has already accepted a different run",
            )
        return {"result": result, "run_id": body.run_id}

    @app.get("/model")
    def model(request: Request) -> Response:
        payload, headers = get_runtime(request).model_snapshot()
        return Response(
            content=payload,
            media_type="application/octet-stream",
            headers=headers,
        )

    @app.get("/status")
    def client_status(request: Request) -> dict[str, object]:
        return get_runtime(request).status()

    return app


app = create_app()

