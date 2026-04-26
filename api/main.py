import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Must be set before any bakesquad imports so llm_client picks up the right backend.
os.environ.setdefault("MODEL_BACKEND", "openai")

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import prefs, recipe_box, score_url, search


@asynccontextmanager
async def lifespan(app: FastAPI):
    from bakesquad.memory import init_db
    init_db()
    yield


app = FastAPI(title="BakeSquad API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # Vite dev server
        "http://localhost:4173",   # Vite preview
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(score_url.router, prefix="/api")
app.include_router(search.router,    prefix="/api")
app.include_router(recipe_box.router, prefix="/api")
app.include_router(prefs.router,     prefix="/api")
