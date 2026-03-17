from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

app = FastAPI(title="NCAA March Madness 2026 Bracket Predictor", version="1.0.0")

from server.routes import predictions, bracket

app.include_router(predictions.router, prefix="/api", tags=["predictions"])
app.include_router(bracket.router, prefix="/api", tags=["bracket"])


@app.get("/api/health")
def health():
    return {"status": "ok", "app": "ncaa-bracket-2026"}


# Serve React frontend in production
frontend_dist = os.path.join(os.path.dirname(__file__), "frontend", "dist")
if os.path.exists(frontend_dist):
    assets_dir = os.path.join(frontend_dist, "assets")
    if os.path.exists(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        index = os.path.join(frontend_dist, "index.html")
        return FileResponse(index)
