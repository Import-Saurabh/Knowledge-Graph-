# News Intelligence Knowledge Graph — Dynamic Ontology

Dynamic, self-expanding knowledge graph from **war reporting** with **zero hardcoded taxonomies**.

Based on your uploaded News Intelligence Knowledge Graph architecture specification, here is a complete end-to-end project data flow showing where every technology is used. 

# High-Level Architecture

```text
                         ┌──────────────────────┐
                         │ Raw News Articles    │
                         │ JSON / JSONL         │
                         └──────────┬───────────┘
                                    │
                                    ▼
                     ┌──────────────────────────┐
                     │ Stage 1: Ingestion       │
                     │ ArticleLoader            │
                     └──────────┬───────────────┘
                                │
                                │ Store Metadata
                                ▼
                      ┌─────────────────────┐
                      │ SQLite Database     │
                      │ SQLAlchemy ORM      │
                      └─────────┬───────────┘
                                │
                                ▼
               ┌─────────────────────────────────┐
               │ Stage 2: Entity Extraction      │
               │ GLiNER / spaCy                  │
               └──────────────┬──────────────────┘
                              │
                              ▼
                    Entity Mentions
                              │
                              ▼
          ┌──────────────────────────────────────┐
          │ Stage 3: Entity Resolution           │
          │ RapidFuzz + Embeddings + ChromaDB    │
          └──────────────┬───────────────────────┘
                         │
                         ▼
              Canonical Entities Created
                         │
                         ├────────────► SQLite
                         │
                         └────────────► ChromaDB

```

---

# Full Pipeline Flow

```text
┌────────────────────────────────────────────────────────────┐
│                     NEWS ARTICLES                          │
└────────────────────────────────────────────────────────────┘
                           │
                           ▼

┌────────────────────────────────────────────────────────────┐
│ Stage 1 : INGESTION                                        │
│ Technology:                                                 │
│ • Python                                                   │
│ • Pydantic                                                 │
│ • SQLAlchemy                                               │
│ • SQLite                                                   │
└────────────────────────────────────────────────────────────┘
                           │
                           ▼

┌────────────────────────────────────────────────────────────┐
│ Stage 2 : ENTITY EXTRACTION                                │
│ Technology:                                                 │
│ • GLiNER                                                   │
│ • spaCy (Fallback)                                         │
│ • Dynamic Ontology Labels                                 │
└────────────────────────────────────────────────────────────┘
                           │
                           ▼

                Entity Mentions
                           │
                           ▼

┌────────────────────────────────────────────────────────────┐
│ Stage 3 : ENTITY RESOLUTION                                │
│ Technology:                                                 │
│ • RapidFuzz                                                │
│ • BGE Embeddings                                           │
│ • ChromaDB Entity Collection                              │
│ • SQLite Alias Store                                      │
└────────────────────────────────────────────────────────────┘
                           │
                           ▼

                Canonical Entities
                           │
                           ▼

┌────────────────────────────────────────────────────────────┐
│ Stage 4 : EMBEDDING GENERATION                             │
│ Technology:                                                 │
│ • Sentence Transformers                                    │
│ • BAAI/bge-small-en-v1.5                                   │
│ • Torch                                                    │
└────────────────────────────────────────────────────────────┘
                           │
                           ▼

                   Article Vectors
                           │
                           ▼

┌────────────────────────────────────────────────────────────┐
│ Stage 5 : VECTOR STORAGE                                   │
│ Technology:                                                 │
│ • ChromaDB                                                 │
│                                                           │
│ Collections:                                               │
│ 1. news_articles                                           │
│ 2. canonical_entities                                      │
└────────────────────────────────────────────────────────────┘
                           │
                           ▼

┌────────────────────────────────────────────────────────────┐
│ Stage 6 : DUPLICATE DETECTION                              │
│ Technology:                                                 │
│ • Cosine Similarity                                        │
│ • Scikit-Learn                                             │
└────────────────────────────────────────────────────────────┘
                           │
                           ▼

              Unique Articles Only
                           │
                           ▼

┌────────────────────────────────────────────────────────────┐
│ Stage 7 : EVENT CLUSTERING                                 │
│ Technology:                                                 │
│ • HDBSCAN                                                  │
│ • Temporal Windowing                                       │
└────────────────────────────────────────────────────────────┘
                           │
                           ▼

                    Event Clusters
                           │
                           ▼

┌────────────────────────────────────────────────────────────┐
│ Stage 8 : EVENT BUILDER                                    │
│ Technology:                                                 │
│ • Python                                                   │
│ • Deterministic Logic                                      │
└────────────────────────────────────────────────────────────┘
                           │
                           ▼

                 Event Context
                           │
                           ▼

┌────────────────────────────────────────────────────────────┐
│ Stage 9 : RELATION EXTRACTION                              │
│ Technology:                                                 │
│ • Claude Haiku                                             │
│ OR                                                        │
│ • GPT-4o-mini                                              │
└────────────────────────────────────────────────────────────┘
                           │
                           ▼

            Free-form Relation Triples
                           │
                           ▼

┌────────────────────────────────────────────────────────────┐
│ Stage 9B : RELATION ONTOLOGY                               │
│ Technology:                                                 │
│ • ChromaDB                                                 │
│ • BGE Embeddings                                           │
│ • HDBSCAN                                                  │
└────────────────────────────────────────────────────────────┘
                           │
                           ▼

           Canonical Relations
                           │
                           ▼

┌────────────────────────────────────────────────────────────┐
│ Stage 9C : ENTITY TYPE INDUCTION                           │
│ Technology:                                                 │
│ • LLM Suggestions                                          │
│ • ChromaDB                                                 │
│ • Embedding Similarity                                     │
│ • SQLite Ontology                                          │
└────────────────────────────────────────────────────────────┘
                           │
                           ▼

           Dynamic Ontology Updated
                           │
                           ▼

┌────────────────────────────────────────────────────────────┐
│ Stage 10 : GRAPH BUILDING                                  │
│ Technology:                                                 │
│ • NetworkX                                                 │
└────────────────────────────────────────────────────────────┘
                           │
                           ▼

            Knowledge Graph
                           │
                           ▼

┌────────────────────────────────────────────────────────────┐
│ Stage 11 : GRAPH ANALYTICS                                 │
│ Technology:                                                 │
│ • NetworkX                                                 │
│ • Centrality Metrics                                       │
│ • Co-occurrence Analysis                                   │
└────────────────────────────────────────────────────────────┘
                           │
                           ▼

┌────────────────────────────────────────────────────────────┐
│ Stage 12 : EXPORT                                           │
│ Technology:                                                 │
│ • PyVis                                                    │
│ • CSV Export                                               │
│ • JSON Export                                              │
└────────────────────────────────────────────────────────────┘
                           │
                           ▼

             graph.html
             nodes.csv
             relationships.csv
             analytics_report.json
             ontology_report.json
```

---

# Storage Architecture

```text
                   ┌──────────────┐
                   │ SQLite       │
                   └──────┬───────┘
                          │
      ┌───────────────────┼─────────────────────┐
      │                   │                     │
      ▼                   ▼                     ▼

 Articles Table   Entity Ontology     Relation Ontology

      │
      ▼

 Canonical Entities

      │
      ▼

 Alias Learning Table
```

---

# Vector Architecture

```text
                 ┌─────────────────┐
                 │ Embeddings      │
                 │ BGE Model       │
                 └────────┬────────┘
                          │
          ┌───────────────┴──────────────┐
          │                              │
          ▼                              ▼

  Article Embeddings             Entity Embeddings

          │                              │
          ▼                              ▼

   Chroma Collection            Chroma Collection

    news_articles              canonical_entities
```

---

# Knowledge Graph Structure

```text
                ┌─────────────┐
                │   Event     │
                └──────┬──────┘
                       │
          PARTICIPATES_IN
                       │
     ┌─────────────────┼─────────────────┐
     │                 │                 │
     ▼                 ▼                 ▼

 Entity A         Entity B         Entity C

     │                 │
     │ ATTACKS         │ SANCTIONS
     │                 │
     ▼                 ▼

 Entity D         Entity E
```

---

# Tech Stack by Responsibility

| Responsibility      | Technology                 |
| ------------------- | -------------------------- |
| News Input          | JSON / JSONL               |
| Validation          | Pydantic                   |
| Database            | SQLite + SQLAlchemy        |
| Entity Extraction   | GLiNER                     |
| Fallback NER        | spaCy                      |
| Embeddings          | BGE Small / Large          |
| Vector DB           | ChromaDB                   |
| Entity Matching     | RapidFuzz                  |
| Deduplication       | Scikit-Learn               |
| Event Clustering    | HDBSCAN                    |
| Relation Extraction | Claude Haiku / GPT-4o-mini |
| Ontology Learning   | ChromaDB + Embeddings      |
| Graph Construction  | NetworkX                   |
| Visualization       | PyVis                      |
| Reporting           | JSON                       |
| Export              | CSV                        |

One architectural concern: your design claims "no hardcoded ontology", but GLiNER still requires labels to extract entities. Starting from a completely empty ontology will produce weak extraction quality. In practice, a minimal seed ontology (Person, Organization, Location, Event) is almost mandatory; otherwise the bootstrap phase becomes unreliable. This is the main weakness in the current architecture.

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
