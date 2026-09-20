"""Diem khoi dong FastAPI: lifespan (warmup), mount router, CORS."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from serving.config import settings
from serving.dependencies import get_llm
from serving.routers import cache, health, recognize, recognize_video, tts, websocket
from serving.utils.logger import get_logger
from serving.utils.warmup import warmup_all

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Dang warmup cac service...")
    await warmup_all()
    yield
    await get_llm().close()


app = FastAPI(title="SignSpeak Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(websocket.router)
app.include_router(recognize.router, prefix="/api")
app.include_router(recognize_video.router, prefix="/api")
app.include_router(tts.router, prefix="/api")
app.include_router(cache.router, prefix="/api")
app.include_router(health.router)
