from fastapi import FastAPI

from lol_commentary_backend.api.health import router as health_router


def create_app() -> FastAPI:
    app = FastAPI(
        title="LoL Commentary AI",
        version="0.1.0",
        description="League of Legends AI commentary backend service",
    )

    app.include_router(health_router)

    return app


app = create_app()
