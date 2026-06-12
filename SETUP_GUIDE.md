# Setup Guide — News Intelligence Knowledge Graph

A complete step-by-step guide to get the project running on your machine.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Clone & Install](#2-clone--install)
3. [Choose Your LLM Provider](#3-choose-your-llm-provider)
4. [Configure Environment](#4-configure-environment)
5. [Run the Pipeline](#5-run-the-pipeline)
6. [View Outputs](#6-view-outputs)
7. [Run Tests](#7-run-tests)
8. [Troubleshooting](#8-troubleshooting)
9. [Daily Production Setup](#9-daily-production-setup)

---

## 1. Prerequisites

| Requirement | Version | Why |
|-------------|---------|-----|
| Python | 3.11+ | Required for Pydantic v2, modern type hints |
| pip | 23+ | Package management |
| Git | any | Cloning the repo |
| RAM | 8GB+ | Embedding models, ChromaDB, clustering |
| Disk | 2GB+ | Models, ChromaDB, SQLite, exports |
| OS | Linux/Mac/Windows WSL | All platforms supported |

**Check your Python version:**
```bash
python --version   # Must be 3.11 or higher
```

If you have Python 3.10 or lower, install Python 3.11+ first:
- **Ubuntu/Debian**: `sudo apt install python3.11 python3.11-venv`
- **Mac**: `brew install python@3.11`
- **Windows**: Download from [python.org](https://python.org)

---

## 2. Clone & Install

### Step 2.1 — Extract the project

```bash
# If you have the zip file
unzip news_kg.zip
cd news_kg

# Or if cloning from a repo
git clone <repo-url>
cd news_kg
```

### Step 2.2 — Create a virtual environment

```bash
# Linux / Mac
python -m venv venv
source venv/bin/activate

# Windows (Command Prompt)
python -m venv venv
venv\Scripts\activate.bat

# Windows (PowerShell)
python -m venv venv
venv\Scripts\Activate.ps1
```

You should see `(venv)` in your terminal prompt.

### Step 2.3 — Upgrade pip

```bash
pip install --upgrade pip setuptools wheel
```

### Step 2.4 — Install dependencies

```bash
pip install -r requirements.txt
```

**This will take 5–15 minutes** because it downloads:
- PyTorch (~2GB)
- Sentence Transformers (BGE embedding models)
- spaCy + GLiNER (NER models)
- ChromaDB, NetworkX, PyVis, HDBSCAN, etc.

**If you get build errors for `hdbscan`:**
```bash
# Ubuntu/Debian
sudo apt install build-essential python3-dev

# Mac
brew install llvm libomp

# Then retry
pip install hdbscan
```

### Step 2.5 — Download spaCy model (optional)

Only needed if you plan to use `--fast` mode (spaCy fallback instead of GLiNER):

```bash
python -m spacy download en_core_web_sm
```

---

## 3. Choose Your LLM Provider

The pipeline **needs an LLM** for relation extraction and type induction. Pick **one** of these free or paid options:

### Option A: Groq — FREE (Recommended)

**Cost:** $0. Fastest inference. No credit card required.

**Models available:**
- `llama-3.1-8b-instant` — fastest, good quality *(default)*
- `llama-3.1-70b-versatile` — better quality, still free
- `mixtral-8x7b-32768` — alternative

**Setup:**
1. Go to [console.groq.com](https://console.groq.com)
2. Sign up with email or Google
3. Create an API key
4. Copy the key (starts with `gsk_`)

### Option B: OpenRouter — FREE TIER

**Cost:** $0 for models with `:free` suffix. Credit card optional.

**Models available:**
- `meta-llama/llama-3.1-8b-instruct:free`
- `google/gemma-2-9b-it:free`
- `mistralai/mistral-7b-instruct:free`

**Setup:**
1. Go to [openrouter.ai](https://openrouter.ai)
2. Sign up
3. Go to Keys → Create Key
4. Copy the key (starts with `sk-or-v1-`)

### Option C: Moonshot AI (Kimi)

**Cost:** Has a free trial tier. Chinese provider.

**Models:**
- `moonshot-v1-8k`
- `moonshot-v1-32k`
- `moonshot-v1-128k`

**Setup:**
1. Go to [platform.moonshot.cn](https://platform.moonshot.cn)
2. Sign up and verify
3. Create API key

### Option D: No LLM (Skip Relation Extraction)

**Cost:** $0. No API key needed at all.

Use this to test the pipeline skeleton. The graph will have events and entities but **no relations** between entities.

---

## 4. Configure Environment

### Step 4.1 — Copy the example env file

```bash
cp .env.example .env
```

### Step 4.2 — Edit `.env` with your provider

**For Groq (free):**
```bash
cat > .env << 'EOF'
GROQ_API_KEY=gsk_your_actual_key_here
LLM_PROVIDER=groq
LLM_MODEL=llama-3.1-8b-instant
EMBEDDING_MODEL=small
USE_SPACY_FALLBACK=false
DATABASE_URL=sqlite:///data/news_kg.db
CHROMA_PERSIST_DIR=data/chroma_db
SEED_ONTOLOGY_PATH=config/seed_ontology.yaml
ENABLE_TYPE_INDUCTION=true
ENABLE_RELATION_CLUSTERING=true
TEMPORAL_WINDOW_DAYS=7
MIN_CLUSTER_SIZE=3
SIMILARITY_THRESHOLD=0.95
ENTITY_MERGE_THRESHOLD=0.88
NER_BATCH_SIZE=32
EMBEDDING_BATCH_SIZE=64
EOF
```

**For OpenRouter (free):**
```bash
cat > .env << 'EOF'
OPENROUTER_API_KEY=sk-or-v1-your_actual_key_here
LLM_PROVIDER=openrouter
LLM_MODEL=meta-llama/llama-3.1-8b-instruct:free
EMBEDDING_MODEL=small
USE_SPACY_FALLBACK=false
DATABASE_URL=sqlite:///data/news_kg.db
CHROMA_PERSIST_DIR=data/chroma_db
SEED_ONTOLOGY_PATH=config/seed_ontology.yaml
ENABLE_TYPE_INDUCTION=true
ENABLE_RELATION_CLUSTERING=true
TEMPORAL_WINDOW_DAYS=7
MIN_CLUSTER_SIZE=3
SIMILARITY_THRESHOLD=0.95
ENTITY_MERGE_THRESHOLD=0.88
NER_BATCH_SIZE=32
EMBEDDING_BATCH_SIZE=64
EOF
```

**For Moonshot:**
```bash
cat > .env << 'EOF'
MOONSHOT_API_KEY=your_actual_key_here
LLM_PROVIDER=moonshot
LLM_MODEL=moonshot-v1-8k
EMBEDDING_MODEL=small
USE_SPACY_FALLBACK=false
DATABASE_URL=sqlite:///data/news_kg.db
CHROMA_PERSIST_DIR=data/chroma_db
SEED_ONTOLOGY_PATH=config/seed_ontology.yaml
ENABLE_TYPE_INDUCTION=true
ENABLE_RELATION_CLUSTERING=true
TEMPORAL_WINDOW_DAYS=7
MIN_CLUSTER_SIZE=3
SIMILARITY_THRESHOLD=0.95
ENTITY_MERGE_THRESHOLD=0.88
NER_BATCH_SIZE=32
EMBEDDING_BATCH_SIZE=64
EOF
```

**For no LLM (testing only):**
```bash
cat > .env << 'EOF'
LLM_PROVIDER=groq
GROQ_API_KEY=dummy
EMBEDDING_MODEL=small
USE_SPACY_FALLBACK=true
DATABASE_URL=sqlite:///data/news_kg.db
CHROMA_PERSIST_DIR=data/chroma_db
SEED_ONTOLOGY_PATH=config/seed_ontology.yaml
ENABLE_TYPE_INDUCTION=true
ENABLE_RELATION_CLUSTERING=true
TEMPORAL_WINDOW_DAYS=7
MIN_CLUSTER_SIZE=3
SIMILARITY_THRESHOLD=0.95
ENTITY_MERGE_THRESHOLD=0.88
NER_BATCH_SIZE=32
EMBEDDING_BATCH_SIZE=64
EOF
```

### Step 4.3 — Verify your env file

```bash
cat .env | grep -v "^#" | grep -v "^$"
```

You should see your API key and provider settings.

---

## 5. Run the Pipeline

### First Run — Full Pipeline (Recommended)

```bash
python main.py --input data/raw/ --run-all
```

**What happens:**
1. Loads 20 sample articles from `data/raw/sample_articles.jsonl`
2. Extracts entities with GLiNER (or spaCy if `--fast`)
3. Resolves entities to canonical forms via embeddings + fuzzy matching
4. Generates article embeddings and stores in ChromaDB
5. Detects and marks duplicate articles
6. Clusters articles into events via HDBSCAN
7. Builds event contexts
8. Calls LLM to extract relations and discover entity types
9. Normalizes relations through the relation ontology
10. Builds the NetworkX knowledge graph
11. Exports to CSV, HTML, and JSON reports

**Expected output:**
```
{"event": "ingestion_complete", "inserted": 20, "skipped": 0, "total": 20}
{"event": "ner_complete", "mentions": 150}
{"event": "resolution_complete", "canonical_entities": 45}
{"event": "events_built", "count": 5}
{"event": "pipeline_complete", "nodes": 52, "edges": 78, "entity_types": 12, "relation_types": 8}
```

### Fast Mode (No LLM, spaCy NER)

Use this for quick testing without any API keys:

```bash
python main.py --input data/raw/ --run-all --fast --skip-llm
```

**Differences:**
- Uses spaCy `en_core_web_sm` instead of GLiNER (faster, less accurate)
- Skips LLM relation extraction entirely
- Graph will have events and entities but no inter-entity relations
- Completes in ~2–3 minutes instead of ~10–15 minutes

### With Custom Articles

Replace `data/raw/sample_articles.jsonl` with your own `.jsonl` file:

```json
{"title": "Your article title", "content": "Your article content...", "source": "Reuters", "published_at": "2024-01-15T10:30:00Z", "url": "https://example.com/article-1"}
{"title": "Another article", "content": "More content...", "source": "BBC", "published_at": "2024-01-16T14:20:00Z", "url": "https://example.com/article-2"}
```

Then run:
```bash
python main.py --input data/raw/ --run-all
```

### Resume / Daily Mode

If the pipeline crashes or you want to process only new articles:

```bash
# Only processes articles not yet in the database
python main.py --input data/raw/daily/ --run-all --daily
```

---

## 6. View Outputs

After the pipeline completes, check `data/exports/`:

```bash
ls -la data/exports/
```

### Interactive Graph

```bash
# Mac
open data/exports/graph.html

# Linux
xdg-open data/exports/graph.html

# Windows
start data/exports/graph.html
```

This opens an interactive PyVis visualization where you can:
- Drag nodes around
- Zoom and pan
- Click nodes to see details (type, mention count)
- Orange nodes = Events, colored nodes = Entities (color = type)

### CSV Files

```bash
# View nodes (entities + events)
head -20 data/exports/nodes.csv

# View relationships
head -20 data/exports/relationships.csv
```

### Analytics Report

```bash
# Pretty-print JSON
cat data/exports/analytics_report.json | python -m json.tool

# Or use jq if installed
cat data/exports/analytics_report.json | jq '.ontology_stats'
```

Key metrics:
- `degree_centrality` — most connected nodes
- `betweenness_centrality` — bridge nodes between events
- `top_entities_by_mentions` — most frequently mentioned entities
- `event_frequency_by_window` — events per week
- `ontology_stats.entity_types_discovered` — how many types the system learned
- `ontology_stats.relation_types_discovered` — how many relation types

### Ontology Report

```bash
cat data/exports/ontology_report.json | python -m json.tool
```

Shows:
- All discovered entity types (e.g., "AI Company", "Central Bank", "Armed Group")
- All discovered relation types (e.g., "ATTACKS", "SANCTIONS", "SIGNED_DEAL_WITH")
- Which types are auto-discovered vs. user-confirmed

---

## 7. Run Tests

```bash
# All tests
pytest tests/ -v

# Specific modules
pytest tests/test_ontology_manager.py -v
pytest tests/test_entity_resolver.py -v
pytest tests/test_graph_builder.py -v
pytest tests/test_relation_ontology.py -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

---

## 8. Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'src'`

**Fix:** Make sure you're running from the project root:
```bash
cd /path/to/news_kg
python main.py --input data/raw/ --run-all
```

### Issue: `chromadb.errors.NoIndexException` or ChromaDB errors

**Fix:** Delete the ChromaDB directory and restart:
```bash
rm -rf data/chroma_db
python main.py --input data/raw/ --run-all
```

### Issue: `sqlite3.OperationalError: no such table`

**Fix:** The database wasn't initialized. Run:
```bash
rm -f data/news_kg.db
python -c "from src.utils.db import init_db; init_db()"
python main.py --input data/raw/ --run-all
```

### Issue: LLM returns invalid JSON / parse errors

**Fix:** This happens with smaller free models. The code retries 3 times automatically. If it keeps failing:
1. Switch to a stronger model (e.g., `llama-3.1-70b-versatile` on Groq)
2. Or use `--skip-llm` to bypass relation extraction

### Issue: `CUDA out of memory` or RAM issues

**Fix:** Use the small embedding model and spaCy fallback:
```bash
# In .env
EMBEDDING_MODEL=small
USE_SPACY_FALLBACK=true
```

Then run with `--fast --skip-llm`.

### Issue: Groq rate limit (429 errors)

**Fix:** The code already adds 1-second delays between LLM calls. If you still hit limits:
1. Reduce batch size: set `MIN_CLUSTER_SIZE=5` in `.env` (fewer events = fewer LLM calls)
2. Or upgrade to Groq's paid tier for higher rate limits

### Issue: `hdbscan` fails to install

**Fix:**
```bash
# Ubuntu/Debian
sudo apt-get install python3-dev build-essential

# Mac
brew install llvm libomp
export CC=/usr/local/opt/llvm/bin/clang

# Then
pip install --no-cache-dir hdbscan
```

### Issue: GLiNER model download fails

**Fix:** Use spaCy fallback:
```bash
python main.py --input data/raw/ --run-all --fast
```

---

## 9. Daily Production Setup

For running this as a daily cron job:

### Step 9.1 — Create a shell script

```bash
cat > run_daily.sh << 'EOF'
#!/bin/bash
set -e

cd /path/to/news_kg
source venv/bin/activate

# Process only new articles from today's folder
python main.py --input data/raw/daily/ --run-all --daily

# Optional: backup exports
timestamp=$(date +%Y%m%d_%H%M%S)
cp data/exports/ontology_report.json "data/processed/ontology_${timestamp}.json"
cp data/exports/analytics_report.json "data/processed/analytics_${timestamp}.json"
cp data/exports/graph.html "data/processed/graph_${timestamp}.html"
EOF

chmod +x run_daily.sh
```

### Step 9.2 — Add to crontab

```bash
crontab -e
```

Add this line to run every day at 23:59:
```cron
59 23 * * * /path/to/news_kg/run_daily.sh >> /path/to/news_kg/data/processed/cron.log 2>&1
```

### Step 9.3 — Monitor

```bash
# Check last run
tail -50 data/processed/cron.log

# Check if exports were generated
ls -lt data/exports/
```

---

## Quick Reference Card

| Command | Purpose | Time | Needs API Key |
|---------|---------|------|---------------|
| `python main.py --input data/raw/ --run-all` | Full pipeline | ~10–15 min | Yes |
| `python main.py --input data/raw/ --run-all --fast` | spaCy NER + LLM | ~5–8 min | Yes |
| `python main.py --input data/raw/ --run-all --skip-llm` | No relations | ~3–5 min | No |
| `python main.py --input data/raw/ --run-all --fast --skip-llm` | Fastest test | ~2–3 min | No |
| `python main.py --input data/raw/ --run-all --daily` | Only new articles | Varies | Yes |
| `pytest tests/ -v` | Run unit tests | ~1 min | No |

---

## File Structure After First Run

```
news_kg/
├── data/
│   ├── raw/
│   │   └── sample_articles.jsonl
│   ├── processed/          # Created on first run
│   ├── exports/            # Created on first run
│   │   ├── graph.html
│   │   ├── nodes.csv
│   │   ├── relationships.csv
│   │   ├── analytics_report.json
│   │   └── ontology_report.json
│   ├── chroma_db/          # Created on first run
│   └── news_kg.db          # Created on first run
├── .env                    # You create this
└── ...
```
