from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app import config
from app.routers import meta, predict

app = FastAPI(
    title=config.APP_NAME,
    version=config.APP_VERSION,
    description=(
        "Cascaded coffee-leaf detection and disease diagnosis API for Ethiopian "
        "smallholder coffee farms, with expert multilingual advisory reports."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router, prefix="/api/v1")
app.include_router(meta.router, prefix="/api/v1")


# ----------------------------------------------------------------------
# Optional: serve the static frontend from the same free web service.
# This lets the whole product (UI + API) run as ONE process on a single
# free-tier host (Render, Railway, Fly.io, Hugging Face Spaces, etc.)
# with no separate frontend hosting, no CORS setup, and no extra cost.
# ----------------------------------------------------------------------
def _resolve_frontend_dir() -> str | None:
    if config.FRONTEND_DIR:
        return config.FRONTEND_DIR if os.path.isdir(config.FRONTEND_DIR) else None
    # default: ../../frontend relative to this file (backend/app/main.py -> frontend/)
    candidate = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "frontend"))
    return candidate if os.path.isdir(candidate) else None


_frontend_dir = _resolve_frontend_dir() if config.SERVE_FRONTEND else None

if _frontend_dir:

    @app.get("/", include_in_schema=False)
    def serve_index():
        return FileResponse(os.path.join(_frontend_dir, "index.html"))

    @app.get("/{path:path}", include_in_schema=False)
    def serve_static(path: str):
        candidate = os.path.join(_frontend_dir, path)
        if os.path.isfile(candidate):
            return FileResponse(candidate)
        return FileResponse(os.path.join(_frontend_dir, "index.html"))
