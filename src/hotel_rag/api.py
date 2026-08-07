"""API REST exposant l'assistant RAG.

Lancer en local :
    uvicorn hotel_rag.api:app --reload

Le modèle et l'index vectoriel sont chargés une seule fois, au démarrage
du serveur (voir `lifespan`), et réutilisés pour chaque requête.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from hotel_rag.pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_state: dict[str, RAGPipeline] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Démarrage : chargement du pipeline RAG (modèles + index)...")
    _state["pipeline"] = RAGPipeline()
    logger.info("Pipeline prêt.")
    yield
    _state.clear()


app = FastAPI(
    title="Hotel RAG Assistant",
    description="Assistant virtuel documentaire pour l'Hôtel Le Belvédère, basé sur un pipeline RAG.",
    version="1.0.0",
    lifespan=lifespan,
)


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, examples=["A quelle heure commence le check-in ?"])
    top_k: int = Field(default=2, ge=1, le=15)


class SourceItem(BaseModel):
    source: str
    title: str
    score: float


class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceItem]


def get_pipeline() -> RAGPipeline:
    pipeline = _state.get("pipeline")
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Le pipeline n'est pas encore prêt.")
    return pipeline


@app.get("/health")
def health() -> dict:
    """Vérifie que le service est démarré et que la documentation est indexée."""
    pipeline = _state.get("pipeline")
    if pipeline is None:
        return {"status": "starting"}
    return {"status": "ok", "documents_indexed": pipeline.num_documents}


@app.post("/ask", response_model=AnswerResponse)
def ask(request: QuestionRequest) -> AnswerResponse:
    """Pose une question à l'assistant et reçoit une réponse sourcée."""
    pipeline = get_pipeline()
    result = pipeline.answer(request.question, top_k=request.top_k)
    return AnswerResponse(
        question=result.question,
        answer=result.answer,
        sources=[SourceItem(**s) for s in result.sources_summary()],
    )
