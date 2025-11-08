# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sqlite3
import os
import httpx
import uuid
from typing import List
from sentence_transformers import SentenceTransformer, util

app = FastAPI(title="Transcript Query Service")

embedder = SentenceTransformer("all-MiniLM-L6-v2")

DB_PATH = "transcripts.db"
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
c = conn.cursor()
c.execute("""
CREATE TABLE IF NOT EXISTS transcripts (
    id TEXT PRIMARY KEY,
    content TEXT,
    embedding BLOB
)
""")
conn.commit()

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5

class SummarizeRequest(BaseModel):
    transcript_id: str

async def fetch_limitless_transcripts(limit: int = 10) -> List[str]:
    api_key = os.getenv("LIMITLESS_API_KEY")
    if not api_key:
        raise RuntimeError("LIMITLESS_API_KEY not set")
    url = "https://api.limitless.ai/v1/lifelogs"
    headers = {"X-API-Key": api_key}
    params = {"limit": limit}
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(url, headers=headers, params=params)
        if resp.status_code != 200:
            raise RuntimeError(f"Limitless API error: {resp.status_code}")
        data = resp.json()
        entries = [entry.get("transcript", "") for entry in data.get("data", [])]
    return entries

def save_transcript(text: str) -> str:
    vector = embedder.encode(text)
    tid = str(uuid.uuid4())
    c.execute("INSERT INTO transcripts (id, content, embedding) VALUES (?, ?, ?)",
              (tid, text, vector.tobytes()))
    conn.commit()
    return tid

def search_transcripts(question: str, top_k: int = 5):
    c.execute("SELECT id, content, embedding FROM transcripts")
    rows = c.fetchall()
    if not rows:
        return []
    ids, contents, embeddings = zip(*rows)
    embeddings = [util.tensor_to_numpy(util.torch_tensor_from_bytes(e)) for e in embeddings]
    query_embedding = embedder.encode(question)
    scores = util.dot_score(query_embedding, embeddings).squeeze()
    top_indices = scores.argsort()[-top_k:][::-1]
    results = [{"id": ids[i], "content": contents[i], "score": float(scores[i])} for i in top_indices]
    return results

def summarize_text(text: str) -> str:
    sentences = text.strip().split(".")
    return ". ".join(sentences[:2]).strip()

@app.post("/ingest_limitless")
async def ingest_limitless():
    try:
        transcripts = await fetch_limitless_transcripts(limit=20)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    ids = []
    for t in transcripts:
        if t.strip():
            ids.append(save_transcript(t))
    return {"ingested": ids}

@app.post("/query")
def query_transcripts(request: QueryRequest):
    results = search_transcripts(request.question, request.top_k)
    return {"results": results}

@app.post("/summarize")
def summarize_transcript(request: SummarizeRequest):
    c.execute("SELECT content FROM transcripts WHERE id = ?", (request.transcript_id,))
    row = c.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Transcript not found")
    summary = summarize_text(row[0])
    return {"id": request.transcript_id, "summary": summary}
