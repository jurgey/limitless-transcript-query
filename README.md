# Limitless Transcript Query

This project provides a FastAPI backend service for processing transcripts from Limitless AI pendant recordings. It ingests transcripts via the Limitless API, stores them in a local SQLite database with embeddings for semantic search, and exposes endpoints for querying and summarization.

## Features

- Ingest latest transcripts from Limitless pendant via API
- Semantic search across transcripts using sentence-transformers
- Summarize transcripts to extract key points
- Generate simple actionable lists (TODO: implement extraction)
- REST endpoints for ingestion (`/ingest_limitless`), query (`/query`), and summarization (`/summarize`)

## Requirements

Dependencies are listed in `requirements.txt`. Install them using pip:

```
pip install -r requirements.txt
```

## Running the Service

Start the FastAPI server with uvicorn:

```
uvicorn app:app --reload
```

Ensure you set the `LIMITLESS_API_KEY` environment variable with your Limitless API key.

## Usage

1. Run the server.
2. POST to `/ingest_limitless` to fetch and store recent transcripts.
3. POST to `/query` with a question to perform semantic search.
4. POST to `/summarize` with a transcript ID to get a summary.

## License

MIT License
