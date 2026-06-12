# News Intelligence Knowledge Graph — Dynamic Ontology

Dynamic, self-expanding knowledge graph from **war reporting** with **zero hardcoded taxonomies**.

> **Current Demo Dataset:** Iran-Israel War of 2026 (Feb 28 – present) — 38 articles covering military operations, sanctions, diplomacy, cyber warfare, humanitarian crisis, and oil market impacts. The system itself is **not hardcoded to Iran** — it will work on any war dataset you provide.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download spaCy model (optional, for fast mode)
python -m spacy download en_core_web_sm

# Run pipeline with FREE Groq model (no credit card needed)
python main.py --input data/raw/ --run-all

# Or skip LLM for fastest testing (no API key at all)
python main.py --input data/raw/ --run-all --skip-llm --fast
```

## What This Prototype Does

1. **Ingests war reporting** (e.g., Iran conflict articles, Ukraine articles, Gaza articles)
2. **Discovers entity types dynamically** — no hardcoded "Person/Organization" lists. The system learns types like:
   - *Militant Group*, *Ballistic Missile*, *Nuclear Facility*, *Defense System*, *Oil Tanker*, *Proxy Force*
3. **Resolves entities** via embeddings + fuzzy matching — learns aliases automatically (e.g., "IRGC" = "Islamic Revolutionary Guard Corps")
4. **Clusters events temporally** — groups articles by week into distinct war events
5. **Extracts relations** via LLM — discovers free-form relations like "launched airstrike against", "imposed sanctions on", "seized tanker"
6. **Normalizes relations** into a learned ontology — clusters similar relations automatically
7. **Builds an interactive graph** — events, entities, and relations with full provenance

## LLM Provider Setup (Free Options)

### 1. Groq — FREE (Recommended, fastest)
Sign up at [console.groq.com](https://console.groq.com) → get free API key → set in `.env`:
```bash
GROQ_API_KEY=gsk_your_key_here
LLM_PROVIDER=groq
LLM_MODEL=llama-3.1-8b-instant   # free, fast
# Or: llama-3.1-70b-versatile, mixtral-8x7b-32768
```

### 2. OpenRouter — FREE TIER
Sign up at [openrouter.ai](https://openrouter.ai) → get free API key → set in `.env`:
```bash
OPENROUTER_API_KEY=sk-or-v1-your_key_here
LLM_PROVIDER=openrouter
LLM_MODEL=meta-llama/llama-3.1-8b-instruct:free
# Or: google/gemma-2-9b-it:free, mistralai/mistral-7b-instruct:free
```

### 3. Moonshot AI (Kimi)
Sign up at [platform.moonshot.cn](https://platform.moonshot.cn) → get API key → set in `.env`:
```bash
MOONSHOT_API_KEY=your_key_here
LLM_PROVIDER=moonshot
LLM_MODEL=moonshot-v1-8k
```

### 4. No LLM (Testing)
```bash
python main.py --input data/raw/ --run-all --skip-llm --fast
```

## Outputs

- `data/exports/graph.html` — Interactive PyVis visualization (events + entities + relations)
- `data/exports/nodes.csv` — Graph nodes (entities + events)
- `data/exports/relationships.csv` — Graph edges (relations)
- `data/exports/analytics_report.json` — Graph metrics (centrality, top entities, event frequency)
- `data/exports/ontology_report.json` — Discovered entity & relation types (auto-learned from war data)

## Architecture

- **Dynamic Ontology**: Entity types and relation types discovered from data via LLM induction
- **Embedding Resolution**: ChromaDB + rapidfuzz for entity canonicalization
- **Zero Hardcoded Taxonomies**: All types learned and stored in SQLite
- **Event-Centric Graph**: NetworkX DiGraph with temporal clustering via HDBSCAN
- **Multi-Provider LLM**: Groq, OpenRouter, Moonshot, Anthropic, OpenAI

## Using Your Own War Data

Replace `data/raw/sample_articles.jsonl` with your own `.jsonl` file:

```json
{"title": "Your war article title", "content": "Full article text...", "source": "Reuters", "published_at": "2026-06-12T10:30:00Z", "url": "https://example.com/article-1"}
```

The system works on **any war** — Ukraine, Gaza, Yemen, Sudan, etc. — without code changes.

## Project Structure

```
news_kg/
├── data/raw/              # Input war articles (.jsonl)
├── data/exports/          # Output CSVs, HTML, JSON
├── config/                # Optional seed ontology
├── src/                   # All pipeline modules (generic, not war-specific)
│   ├── models/            # Pydantic models
│   ├── ingestion/         # Article loading
│   ├── entities/          # NER, resolution, ontology (dynamic)
│   ├── embeddings/        # BGE embeddings
│   ├── vectorstore/       # ChromaDB manager
│   ├── deduplication/     # Duplicate detection
│   ├── clustering/        # HDBSCAN event clustering
│   ├── events/            # Event context builder
│   ├── relations/         # LLM extraction + relation ontology
│   ├── graph/             # NetworkX builder + PyVis export
│   ├── analytics/         # Graph metrics
│   └── utils/             # Config, logger, DB
├── tests/                 # pytest suite
├── main.py                # Pipeline orchestrator
└── requirements.txt
```

## Running Tests

```bash
pytest tests/ -v
```

## Daily Operation

```bash
# Cron job at 23:59 every day
python main.py --input data/raw/daily/ --run-all --daily
```
