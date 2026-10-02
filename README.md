# LogIQ — Enterprise Log Intelligence RCA Engine

An agentic Root Cause Analysis system for multi-tenant production environments. Accepts natural-language incident queries, retrieves evidence from a hybrid SQL + semantic vector index, validates citations deterministically, and returns grounded, causal-chain-ordered RCA reports.

---

## What It Does

Given a production log stream from multiple tenants and a natural-language query like:

> "What caused the 503 errors for TENANT-X around 12:03?"

LogIQ returns:

```json
{
  "root_cause": "Database connection pool exhaustion on DB-Cluster-01",
  "confidence": 0.87,
  "causal_chain": [
    "DB connection pool utilization exceeded 90% threshold on DB-Cluster-01",
    "pool.ConnectionTimeoutException thrown after 3000ms wait for idle connection",
    "CircuitBreaker transitioned CLOSED → OPEN for DB-Cluster-01",
    "503 Service Unavailable cascade — all TENANT-X traffic rejected"
  ],
  "evidence_sources": [
    {
      "timestamp": "2026-05-26T12:02:55.540",
      "level": "WARN",
      "component": "DataSource",
      "message": "Database connection pool utilization exceeding 90% for cluster [DB-Cluster-01]."
    },
    {
      "timestamp": "2026-05-26T12:02:58.378",
      "level": "ERROR",
      "component": "DataSource",
      "message": "pool.ConnectionTimeoutException: Timeout waiting for idle connection after 3000ms."
    }
  ],
  "recommended_fix": "Increase DB connection pool size. Add a 30s query timeout on DataSource. Consider a read replica to distribute sustained load.",
  "duration_ms": 3847.2,
  "iteration_count": 1
}
```

The system correctly surfaces the 2-line root cause cluster above the 237-line 503 symptom flood using RCA scoring rather than frequency ranking.

---

## Architecture

```
┌──────────────────── LAYER 1: DETERMINISTIC INGESTION ────────────────────┐
│                                                                           │
│  Raw Logs → aiofiles reader → LogParserFactory → LogNormalizer           │
│                ↓ (async, chunked, Semaphore-bounded)                     │
│           SQLite (structured index)    ChromaDB (per-tenant embeddings)  │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────── LAYER 2: AGENTIC DIAGNOSTIC LOOP ────────────────────┐
│                                                                           │
│  POST /query                                                              │
│    → PLANNER   [Gemini]  — NL query → time window + keywords             │
│    → RETRIEVER [hybrid]  — SQL + ChromaDB → RCA-ranked evidence (top 50) │
│    → HYPOTHESIS [Gemini] — top 20 evidence → 2-3 root cause hypotheses   │
│    → EVIDENCE  [Python]  — citation validation, no LLM calls             │
│    → SYNTHESIZER [Gemini + grounding] — causal chain + evidence_sources  │
│    → HTTP response                                                        │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites

- Python 3.9+
- A Gemini API key (get one at [aistudio.google.com](https://aistudio.google.com))

### Install and Run

```bash
# 1. Enter the project directory
cd logiq/

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env and set:
#   GEMINI_API_KEY=your_key_here

# 4. Start the server
uvicorn main:app --reload --port 8000
```

On startup, both seed log files are automatically ingested into SQLite and ChromaDB. No manual ingestion step is required.

Watch the logs — you'll see:
```
Ingesting seed file: auth_rate_limit_noise
Seed ingestion complete [auth_rate_limit_noise]: inserted=5002 skipped=0 tenants=...
Ingesting seed file: production_incident_01
Seed ingestion complete [production_incident_01]: inserted=5003 skipped=0 tenants=...
LangGraph RCA workflow ready.
LogIQ ready.
```

---

## API Reference

### `GET /health`

Checks SQLite connectivity and ChromaDB readiness.

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "db": "connected",
  "vector_store": "ready (8 tenant collections)",
  "version": "0.1.0"
}
```

---

### `POST /ingest`

Ingest a log file into SQLite and ChromaDB. **Idempotent** — re-ingesting the same file skips already-stored records and returns the exact skip count.

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "../production_incident_01.log",
    "file_source_name": "production_incident_01"
  }'
```

```json
{
  "total_lines": 5003,
  "records_inserted": 5003,
  "records_skipped": 0,
  "tenants_found": ["TENANT-A", "TENANT-B", "TENANT-C", "TENANT-D", "TENANT-E", "TENANT-F", "TENANT-X"],
  "duration_seconds": 3.847
}
```

Re-ingesting (idempotency test):
```bash
# Second call — all records already exist, zero duplicates created
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"file_path": "../production_incident_01.log"}'
```

```json
{
  "records_inserted": 0,
  "records_skipped": 5003
}
```

---

### `POST /query`

Run a natural-language RCA query scoped to a specific tenant.

**Request schema:**
```json
{
  "query": "string (min 5 chars) — natural-language incident description",
  "tenant_id": "string — must match TENANT-[A-Z]"
}
```

**Scenario 1: Causal Chain Extraction**

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What caused the 503 errors for TENANT-X around 12:03?",
    "tenant_id": "TENANT-X"
  }'
```

Expected response: `root_cause` identifies database connection pool exhaustion. `causal_chain` orders events from the WARN pool utilization lines → ConnectionTimeoutException → CircuitBreaker OPEN → 503 flood. The 237 identical 503 lines are in the evidence but ranked below the root cause.

**Scenario 2: Noise Demultiplexing**

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Identify any system failures impacting TENANT-Z during the authentication volume spike",
    "tenant_id": "TENANT-Z"
  }'
```

Expected response: TENANT-Y's 2,399-line rate-limit storm has zero influence. TENANT-Z's 201 lines are retrieved in isolation. If no failures affected TENANT-Z, the system reports that with high confidence.

---

## Project Structure

```
LogIQ/
├── auth_rate_limit_noise.log      ← seed log: 8 tenants, TENANT-Y rate-limit storm
├── production_incident_01.log     ← seed log: 7 tenants, TENANT-X DB pool failure
│
└── logiq/
    ├── main.py                    ← FastAPI app + lifespan (startup ingestion, graph init)
    ├── config.py                  ← pydantic-settings, reads .env
    ├── graph.py                   ← LangGraph StateGraph assembly + routing logic
    ├── requirements.txt
    ├── .env.example
    │
    ├── api/
    │   ├── routes.py              ← /health, /ingest, /query handlers
    │   └── schemas.py             ← Pydantic request/response models
    │
    ├── agents/
    │   ├── state.py               ← RCAState TypedDict (16 fields)
    │   ├── utils.py               ← parse_llm_json(), build_llm()
    │   ├── planner.py             ← Gemini call #1: NL → search params
    │   ├── retriever.py           ← hybrid SQL + ChromaDB + RCA scoring
    │   ├── hypothesis.py          ← Gemini call #2: evidence → hypotheses
    │   ├── evidence.py            ← deterministic citation validation (no LLM)
    │   └── synthesizer.py         ← Gemini call #3 + evidence grounding
    │
    ├── ingestion/
    │   ├── parser.py              ← LogParserFactory + StandardLogParser (6 regexes)
    │   ├── normalizer.py          ← LogNormalizer (level aliases, whitespace)
    │   └── pipeline.py            ← async chunked ingestion pipeline
    │
    └── storage/
        ├── database.py            ← SQLAlchemy async engine + session factory
        ├── models.py              ← LogRecord (17 fields) + IngestionRun ORM models
        └── vector_store.py        ← ChromaDB wrapper (per-tenant collections)
```

---

## Key Design Decisions

### Hallucination Prevention (Two Layers)

**Layer 1 — Citation Validation (EvidenceAgent):**
The Hypothesis agent asks Gemini to cite supporting log lines by `ingestion_id`. The Evidence agent validates every cited ID against the actual retrieved evidence set (O(1) dict lookup). IDs that don't exist in retrieved evidence are silently dropped. `evidence_strength = found / claimed`. A hypothesis claiming 5 supporting lines but only 1 valid gets `confidence × 0.2`.

**Layer 2 — Evidence Grounding (SynthesizerAgent):**
After Gemini writes the final RCA JSON, the code unconditionally overwrites `evidence_sources` with actual retrieved log lines from the validated hypothesis. Gemini is instructed to return `evidence_sources: []`. This makes it structurally impossible for fabricated log lines to appear in the API response.

### Tenant Isolation (Structural, Not Heuristic)

Every SQL query places `WHERE tenant_id = X` as the first `AND` condition — never optional, never bypassed. ChromaDB collections are named `logs_{tenant_id}`. The `search()` method opens only the collection for the queried tenant. TENANT-Y's 2,399 embeddings cannot influence a TENANT-Z nearest-neighbour search because they live in a separate HNSW graph.

### RCA Scoring (Root Cause vs Symptom Inversion)

Naive frequency-based ranking buries root causes under symptom floods. RCA scoring inverts this:
- `exception_type` present: **+30**
- `circuit_breaker_state` present: **+20**
- `ERROR` level: **+20**
- `WARN` level: **+10**
- `INFO` level: **-10**
- Repetitive message (appears >10x): **-5**

A `ConnectionTimeoutException` line scores 100. The 237 identical 503 lines score 65 each and are de-ranked by the repetition penalty. Root cause surfaces first.

### Infinite Loop Prevention

`max_iterations = 3` is set in the initial `RCAState` and checked first in `should_continue_or_end`. After 3 Evidence agent iterations, the graph unconditionally routes to Synthesizer regardless of the confidence level. This is the hard cap that prevents the LangGraph conditional edge from cycling forever on low-quality evidence.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | (required for /query) | Google Gemini API key |
| `DATABASE_URL` | `sqlite+aiosqlite:///./logiq.db` | SQLAlchemy async database URL |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | ChromaDB persistent storage directory |
| `LOG_CHUNK_SIZE` | `1000` | Lines per ingestion chunk |
| `MAX_CONCURRENT_INGESTION` | `10` | Max parallel parse threads per chunk |

---

## Production Considerations

This is a prototype. For production at enterprise scale:

| Component | Current | Production Path |
|---|---|---|
| Relational DB | SQLite + aiosqlite | PostgreSQL + TimescaleDB + asyncpg |
| Vector DB | ChromaDB local | Pinecone (managed) or Weaviate (self-hosted) |
| Ingestion | Single-process async | Celery workers + Redis broker |
| Observability | Python logging | OpenTelemetry traces + Prometheus |
| Auth | None | JWT per-tenant, API key rotation |
| LLM streaming | Batch `ainvoke` | `astream_events` with Server-Sent Events |

The storage interface (`VectorStore.upsert_logs / search`) and relational models are deliberately isolated. Swapping SQLite → PostgreSQL requires only a `DATABASE_URL` change and one line in the insert logic. Swapping ChromaDB → Pinecone requires implementing the same `upsert_logs / search / list_tenants` interface.
