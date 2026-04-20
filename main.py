"""
Main FastAPI application — mounts RAG and ReAct routers.

Start:
    uvicorn main:app --reload

Endpoints:
    /rag/*    — existing RAG pipeline (CitationQueryEngine + Mem0)
    /react/*  — ReAct agent (AsyncNativeReAct + native tool use)
    /health   — health check
    /         — redirect to docs
"""

import os

os.environ.setdefault("AWS_REGION", "us-east-1")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from rag_app.chat_api import router as rag_router
from react_app.chat_api import router as react_router

app = FastAPI(
    title="Veris Chat API",
    description="Document-grounded conversational system with RAG and ReAct agent modes",
    version="2.0.0",
)

app.include_router(rag_router, prefix="/rag", tags=["RAG"])
app.include_router(react_router, prefix="/react", tags=["ReAct"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/")
async def root():
    return RedirectResponse(url="/docs")
