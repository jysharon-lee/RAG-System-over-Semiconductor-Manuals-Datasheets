import sys
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add the src directory to the python path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from generate_answer import get_answer_data
from hybrid_query import load_retrieval_backend

app = FastAPI(title="Semiconductor RAG API")

# Allow CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the backend globally so we don't reload it on every request
print("Initializing retrieval backend (this may take a moment)...")
backend = load_retrieval_backend()
print("Backend initialized.")


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


@app.post("/api/chat")
async def chat(request: QueryRequest):
    print(f"Received query: {request.question}")
    data = get_answer_data(backend, request.question, top_k=request.top_k)
    return data


@app.get("/api/health")
async def health():
    return {"status": "ok"}
