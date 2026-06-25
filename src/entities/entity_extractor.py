"""
src/entities/entity_extractor.py

GLiNER for named-entity recognition.
GLiREL (jackboyla/glirel-large-v0) for relation extraction.

No spaCy. No LLM. No GLiNER-as-relation-classifier fallback.

If GLiREL cannot load (network down, model not cached), extract_with_glirel()
returns an empty list and logs a clear error — it does NOT fall back to a
degraded approximation. Fix the environment and rerun.

Design
------
  • GLiNER (urchade/gliner_large-v2.1) is loaded once and reused for both
    NER and as the entity-span source for GLiREL.
  • GLiREL receives token-index spans converted from GLiNER's character offsets
    via _tokenize / _char_to_token_span.
  • Relation scoring is fanned across up to `glirel_workers` threads so
    otherwise-idle CPU cores are used during the forward pass.
  • Entity types in RELATION_ENTITY_EXCLUDE are stripped before RE to avoid
    noisy date/money pairs being scored.
"""

import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Tuple, Any

import torch
from src.models.article import ArticleModel
from src.models.entity import EntityMention
from src.utils.config import settings
from src.utils.logger import get_logger

torch.set_num_threads(8)  # reduce inside extract_with_glirel during fan-out

log = get_logger(__name__)

BATCH_SIZE           = 32
CONFIDENCE_THRESHOLD = 0.4

# ---------------------------------------------------------------------------
# GLiNER / GLiREL shared configuration
# ---------------------------------------------------------------------------

MAX_ENTS  = 40   # cap entities per chunk fed to GLiREL
MAX_WORDS = 400  # words per text chunk

# Entity types that add noise to relation pairs (dates, monetary values)
RELATION_ENTITY_EXCLUDE = {"date", "money or economic value"}

TOKEN_RE = re.compile(r"\w+(?:[-_]\w+)*|\S")

# Entity type groups used in CONSTRAINTS
_PER    = {"person"}
_ROLE   = {"job title or role"}
_NATION = {"country", "geopolitical entity"}
_ORG    = {"organization", "government agency", "military unit", "political group"}
_PLACE  = {"country", "city", "location", "geopolitical entity", "facility"}
_ARMS   = {"weapon", "military operation", "vehicle or aircraft"}
_EVENT  = {"event", "military operation"}
_ACTOR  = _PER | _ORG | _NATION
_HITTABLE = _PLACE | _ORG | _PER | _ARMS

# Relation labels for GLiREL — closed-set geopolitical/military vocabulary
RELATION_LABELS: List[str] = [
    "allied with", "at war with", "attacked", "invaded", "located in",
    "capital of", "part of", "member of", "leader of", "head of state of",
    "headquartered in", "supported by", "funded by", "sanctioned by",
    "negotiated with", "signed agreement with", "killed", "based in",
    "occurred in", "controlled by", "founded by", "spokesperson of",
    "citizen of", "participant in", "targeted", "responded to",
    "threatened", "supplied weapons to", "accused",
]

RELATION_CONSTRAINTS: Dict[str, Tuple[set, set]] = {
    "allied with":           (_ACTOR, _ACTOR),
    "at war with":           (_ACTOR, _ACTOR),
    "attacked":              (_ACTOR | _ARMS, _HITTABLE),
    "invaded":               (_ACTOR, _PLACE),
    "located in":            (_PLACE | _ORG | _EVENT, _PLACE),
    "capital of":            ({"city"}, _NATION),
    "part of":               (_ORG | _PLACE, _ORG | _NATION),
    "member of":             (_ACTOR, _ORG | _NATION | {"treaty or agreement"}),
    "leader of":             (_PER | _ROLE, _ACTOR),
    "head of state of":      (_PER, _NATION),
    "headquartered in":      (_ORG, _PLACE),
    "supported by":          (_ACTOR, _ACTOR),
    "funded by":             (_ACTOR, _ACTOR),
    "sanctioned by":         (_ACTOR | _PLACE, _ACTOR),
    "negotiated with":       (_ACTOR, _ACTOR),
    "signed agreement with": (_ACTOR, _ACTOR),
    "killed":                (_ACTOR | _ARMS | _EVENT, _PER),
    "based in":              (_ORG | _PER, _PLACE),
    "occurred in":           (_EVENT, _PLACE),
    "controlled by":         (_PLACE | _ORG | _ARMS, _ACTOR),
    "founded by":            (_ORG | _NATION, _PER | _ORG | _NATION),
    "spokesperson of":       (_PER | _ROLE, _ACTOR),
    "citizen of":            (_PER, _NATION),
    "participant in":        (_ACTOR, _EVENT | {"treaty or agreement"}),
    "targeted":              (_ACTOR | _ARMS, _HITTABLE),
    "responded to":          (_ACTOR, _ACTOR | _EVENT),
    "threatened":            (_ACTOR, _ACTOR),
    "supplied weapons to":   (_ACTOR, _ACTOR),
    "accused":               (_ACTOR, _ACTOR),
}


# ---------------------------------------------------------------------------
# Token-level helpers (needed to convert GLiNER char spans → GLiREL tok spans)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
    toks, spans = [], []
    for m in TOKEN_RE.finditer(text):
        toks.append(m.group())
        spans.append((m.start(), m.end()))
    return toks, spans


def _char_to_token_span(
    c_start: int, c_end: int, spans: List[Tuple[int, int]]
) -> Optional[Tuple[int, int]]:
    first = last = None
    for i, (s, e) in enumerate(spans):
        if e > c_start and s < c_end:
            if first is None:
                first = i
            last = i
    return (first, last) if first is not None else None


def _add_relation(relations: dict, head: str, label: str, tail: str, score: float) -> None:
    """Keep one directed edge per unordered (head, label, tail) — higher score wins."""
    rev = (tail, label, head)
    if rev in relations:
        if score > relations[rev]:
            del relations[rev]
            relations[(head, label, tail)] = score
        return
    key = (head, label, tail)
    if key not in relations or score > relations[key]:
        relations[key] = score


def _passes_constraint(head_type: str, tail_type: str, relation: str) -> bool:
    """Return True if the (head_type, tail_type) pair is allowed for this relation."""
    rule = RELATION_CONSTRAINTS.get(relation)
    if rule is None:
        return True
    return (head_type in rule[0]) and (tail_type in rule[1])


def _batch_predict(
    model, texts: List[str], labels: List[str], threshold: float
) -> List[List[dict]]:
    """
    GLiNER batch helper. Uses batch_predict_entities() when available,
    otherwise loops predict_entities() per text.
    """
    if hasattr(model, "batch_predict_entities"):
        return model.batch_predict_entities(texts, labels, threshold=threshold)
    return [model.predict_entities(t, labels, threshold=threshold) for t in texts]


def _progress(iterable, total: Optional[int] = None, desc: str = ""):
    """
    Yield items from iterable with a tqdm progress bar when available,
    or periodic structured log lines (~20 updates) otherwise.
    """
    try:
        from tqdm import tqdm as _tqdm
        for item in _tqdm(iterable, total=total, desc=desc):
            yield item
        return
    except ImportError:
        pass

    start = time.time()
    step = max(1, total // 20) if total else 50
    for i, item in enumerate(iterable, 1):
        yield item
        if i % step == 0 or (total and i == total):
            elapsed = time.time() - start
            rate = i / elapsed if elapsed > 0 else 0
            eta = (total - i) / rate if (total and rate > 0) else None
            log.info(
                f"{desc}_progress",
                done=i, total=total,
                pct=round(100 * i / total, 1) if total else None,
                elapsed_s=round(elapsed, 1),
                eta_s=round(eta, 1) if eta is not None else None,
            )


# ---------------------------------------------------------------------------
# Real GLiREL relation extractor
# ---------------------------------------------------------------------------

class _RealGlirelExtractor:
    """
    Thin wrapper around glirel.GLiREL (Boylan et al., 2025).

    Requires: pip install "huggingface_hub==0.24.6" glirel --break-system-packages
    (newer huggingface_hub dropped the proxies/resume_download kwargs that
    glirel's _from_pretrained still declares as required — the shim below
    patches them in at runtime so any hub version works).

    Use glirel_base for lower RAM/CPU cost, glirel-large-v0 for higher recall.
    """

    MODEL_NAME = "jackboyla/glirel-large-v0"

    def __init__(self, model_name: Optional[str] = None):
        import inspect
        from glirel import GLiREL

        # Patch missing kwargs for newer huggingface_hub versions
        try:
            raw = GLiREL._from_pretrained
            fn  = raw.__func__ if hasattr(raw, "__func__") else raw
            sig = inspect.signature(fn)
            params = sig.parameters
            needs_patch = any(
                name in params and params[name].default is inspect.Parameter.empty
                for name in ("proxies", "resume_download")
            )
            if needs_patch:
                _orig_fn = fn

                @classmethod  # type: ignore[misc]
                def _patched(cls, *args, proxies=None, resume_download=False, **kwargs):
                    return _orig_fn(
                        cls, *args,
                        proxies=proxies,
                        resume_download=resume_download,
                        **kwargs,
                    )

                GLiREL._from_pretrained = _patched  # type: ignore[method-assign]
        except Exception:
            pass

        self.model = GLiREL.from_pretrained(model_name or self.MODEL_NAME)

    def score_chunk(
        self,
        chunk_text: str,
        gliner_ents: List[dict],
        relation_labels: List[str],
        threshold: float,
    ) -> List[Dict[str, Any]]:
        """
        Score relations between all entity pairs in one text chunk.

        gliner_ents — GLiNER entity dicts with character-offset 'start'/'end'.
        Returns [{head, tail, relation, score}, ...] above threshold.
        """
        if len(gliner_ents) < 2:
            return []

        tokens, spans = _tokenize(chunk_text)
        if not tokens:
            return []

        # Convert char offsets → token spans (GLiREL expects inclusive tok idx)
        ner = []
        for ent in gliner_ents:
            tok_span = _char_to_token_span(ent["start"], ent["end"], spans)
            if tok_span is None:
                continue
            ner.append([tok_span[0], tok_span[1], ent.get("label", "Unknown"), ent["text"]])

        if len(ner) < 2:
            return []

        try:
            relations = self.model.predict_relations(
                tokens, relation_labels, threshold=threshold, ner=ner, top_k=1,
            )
        except Exception as e:
            log.warning("glirel_predict_relations_failed", error=str(e))
            return []

        out: List[Dict[str, Any]] = []
        for r in relations:
            head  = " ".join(r.get("head_text") or []).strip()
            tail  = " ".join(r.get("tail_text") or []).strip()
            label = r.get("label", "")
            score = float(r.get("score", 0.0))
            if not head or not tail or not label or head.lower() == tail.lower():
                continue
            out.append({"head": head, "tail": tail, "relation": label, "score": score})
        return out


# ---------------------------------------------------------------------------
# Main extractor class
# ---------------------------------------------------------------------------

class EntityExtractor:
    """
    Two-stage extractor:
      1. GLiNER NER  — extract named entities from each article.
      2. GLiREL RE   — score relations between discovered entity pairs.

    No spaCy. No LLM. If GLiREL is unavailable, relation extraction returns
    an empty list (no degraded fallback).
    """

    def __init__(
        self,
        ontology_manager=None,
        use_glirel: bool = True,
        glirel_threshold: float = 0.45,
        ner_threshold: float = CONFIDENCE_THRESHOLD,
        glirel_workers: int = 4,
    ):
        self.ontology         = ontology_manager
        self.use_glirel       = use_glirel
        self.glirel_threshold = glirel_threshold
        self.ner_threshold    = ner_threshold
        # i5-1240P: 4 P-cores + 8 E-cores — 4 workers × ~2-3 intra-op threads
        # each is a good starting point. Set glirel_workers=1 if you hit
        # weirdness under concurrency.
        self.glirel_workers   = max(1, glirel_workers)

        self._gliner_model: Optional[object]            = None
        self._glirel_model: Optional[_RealGlirelExtractor] = None
        self._glirel_unavailable: bool                  = False

    # ------------------------------------------------------------------
    # Model loaders
    # ------------------------------------------------------------------

    def _load_gliner(self):
        if self._gliner_model is None:
            try:
                from gliner import GLiNER
                self._gliner_model = GLiNER.from_pretrained(
                    "urchade/gliner_large-v2.1"
                )
                log.info("gliner_model_loaded", model="urchade/gliner_large-v2.1")
            except Exception as e:
                log.error("gliner_load_failed", error=str(e))
                raise
        return self._gliner_model

    def _load_glirel(self) -> Optional[_RealGlirelExtractor]:
        """
        Lazily load GLiREL. Returns None (logged once) if unavailable —
        caller should return empty results rather than running a fallback.

        Load order:
          1. Local HF cache (HF_HUB_OFFLINE=1 — no network ping).
          2. Network download, up to 3 retries with back-off.
        """
        if self._glirel_unavailable:
            return None
        if self._glirel_model is not None:
            return self._glirel_model

        # 1) Try local cache first — avoids failing on a transient DNS blip
        #    when the model is already fully downloaded.
        prev_offline = os.environ.get("HF_HUB_OFFLINE")
        os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            self._glirel_model = _RealGlirelExtractor()
            log.info(
                "glirel_model_loaded",
                model=_RealGlirelExtractor.MODEL_NAME,
                source="local_cache",
            )
            return self._glirel_model
        except Exception:
            pass  # not cached — fall through to network load
        finally:
            if prev_offline is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = prev_offline

        # 2) Network load with retries
        last_err: Optional[Exception] = None
        for attempt in range(1, 4):
            try:
                self._glirel_model = _RealGlirelExtractor()
                log.info(
                    "glirel_model_loaded",
                    model=_RealGlirelExtractor.MODEL_NAME,
                    source="network",
                    attempt=attempt,
                )
                return self._glirel_model
            except Exception as e:
                last_err = e
                if attempt < 3:
                    log.warning("glirel_load_retry", attempt=attempt, error=str(e))
                    time.sleep(3 * attempt)

        self._glirel_unavailable = True
        err_str = str(last_err)
        is_network = any(
            s in err_str for s in (
                "getaddrinfo failed", "NameResolutionError", "ConnectionError",
                "Max retries exceeded", "NewConnectionError", "ConnectTimeout",
            )
        )
        if is_network:
            hint = (
                "GLiREL failed to load due to a network/DNS error. "
                "The model is not yet cached locally — run once with a stable "
                "connection to download it, then it will load offline. "
                "Relation extraction disabled for this run."
            )
        else:
            hint = (
                "GLiREL failed to load. Check: (1) glirel is installed "
                "(pip install glirel --break-system-packages), "
                "(2) huggingface_hub==0.24.6 is pinned, "
                "(3) the local cache isn't corrupted. "
                "Relation extraction disabled for this run."
            )
        log.error("glirel_load_failed", error=err_str, hint=hint)
        return None

    # ------------------------------------------------------------------
    # Label helpers
    # ------------------------------------------------------------------

    def get_gliner_labels(self) -> List[str]:
        if self.ontology:
            return self.ontology.get_active_labels(limit=50)
        return [
            "person", "organization", "country", "city", "location",
            "geopolitical entity", "government agency", "military unit",
            "weapon", "military operation", "event", "treaty or agreement",
            "political group", "nationality", "religion", "ethnic group",
            "facility", "vehicle or aircraft", "job title or role",
            "date", "money or economic value", "law or sanction",
        ]

    # ------------------------------------------------------------------
    # Public NER API
    # ------------------------------------------------------------------

    def extract_batch(
        self, articles: List[ArticleModel]
    ) -> List[List[EntityMention]]:
        """Run GLiNER NER over a batch of articles."""
        try:
            return self._extract_gliner_batch(articles)
        except Exception as e:
            log.error("gliner_extraction_failed", error=str(e))
            raise

    def extract_single(self, article: ArticleModel) -> List[EntityMention]:
        """Run GLiNER NER on a single article."""
        try:
            return self._extract_gliner_batch([article])[0]
        except Exception as e:
            log.warning("gliner_single_failed", error=str(e))
            return []

    # ------------------------------------------------------------------
    # Public RE API
    # ------------------------------------------------------------------

    def extract_with_glirel(
        self, articles: List[ArticleModel]
    ) -> List[Dict]:
        """
        Run GLiNER NER then GLiREL relation extraction over the entity pairs.

        Returns a flat list of dicts: {head, relation, tail, score, article_id}.
        Returns [] if use_glirel=False or if GLiREL cannot load.

        Steps:
          1. Chunk each article into MAX_WORDS windows.
          2. Batch-NER all chunks with GLiNER (sub-batches of BATCH_SIZE).
          3. For each chunk with ≥2 entities, call GLiREL.score_chunk().
             Fanned across glirel_workers threads (PyTorch releases the GIL
             during its CPU kernels, so threads really do run in parallel).
          4. Deduplicate per article via _add_relation() (higher score wins).
        """
        if not self.use_glirel:
            return []

        gliner = self._load_gliner()
        labels = self.get_gliner_labels()
        all_triples: List[Dict] = []

        # ── Step 1: chunk articles ───────────────────────────────────────────
        chunk_to_article: List[Tuple[str, str]] = []  # (article_id, chunk_text)
        for article in articles:
            text = (article.content or "").strip()
            if not text:
                continue
            for chunk in self._chunk_text(text):
                chunk_to_article.append((article.id, chunk))

        if not chunk_to_article:
            return []

        chunk_texts = [chunk for _, chunk in chunk_to_article]

        # ── Step 2: batch NER ────────────────────────────────────────────────
        all_ents_raw: List[List[dict]] = []
        n_sub_batches = (len(chunk_texts) + BATCH_SIZE - 1) // BATCH_SIZE
        ner_t0 = time.time()
        for start in _progress(
            range(0, len(chunk_texts), BATCH_SIZE),
            total=n_sub_batches, desc="gliner_ner",
        ):
            sub_batch = chunk_texts[start:start + BATCH_SIZE]
            try:
                all_ents_raw.extend(
                    _batch_predict(gliner, sub_batch, labels, self.ner_threshold)
                )
            except Exception as e:
                log.error("gliner_batch_ner_failed", batch_start=start, error=str(e))
                all_ents_raw.extend([[] for _ in sub_batch])
        log.info(
            "gliner_ner_complete",
            chunks=len(chunk_texts),
            elapsed_s=round(time.time() - ner_t0, 1),
        )

        # ── Step 3: build work items (chunks with ≥2 usable entities) ───────
        article_chunk_ents: Dict[str, List[Tuple[str, List[dict]]]] = {
            a.id: [] for a in articles
        }
        for (article_id, chunk_text), ents in zip(chunk_to_article, all_ents_raw):
            if ents:
                article_chunk_ents[article_id].append((chunk_text, ents))

        work_items: List[Tuple[str, str, List[dict]]] = []  # (article_id, chunk, ents)
        for article in articles:
            for chunk_text, ents_raw in article_chunk_ents.get(article.id, []):
                ents = [
                    e for e in ents_raw
                    if e["label"].lower() not in RELATION_ENTITY_EXCLUDE
                ][:MAX_ENTS]
                if len(ents) >= 2:
                    work_items.append((article.id, chunk_text, ents))

        # ── Load GLiREL ──────────────────────────────────────────────────────
        log.info(
            "glirel_loading",
            total_chunks=len(work_items),
            hint="Loading GLiREL — ~10-30s first call, instant from cache after...",
        )
        glirel = self._load_glirel()
        if glirel is None:
            log.error(
                "glirel_unavailable",
                hint="Relation extraction skipped — fix GLiREL load error above and rerun.",
            )
            return []

        # ── Step 4: score relations ──────────────────────────────────────────
        per_article_relations: Dict[str, Dict[Tuple, float]] = {
            a.id: {} for a in articles
        }
        n_workers    = self.glirel_workers
        prev_threads = torch.get_num_threads()
        torch.set_num_threads(max(1, prev_threads // n_workers))
        re_t0 = time.time()
        try:
            if n_workers == 1:
                _prev_aid  = None
                _aid_idx   = 0
                _aid_order = list(dict.fromkeys(aid for aid, _, _ in work_items))
                _n_arts    = len(_aid_order)
                for article_id, chunk_text, ents in _progress(
                    work_items, total=len(work_items), desc="glirel_score"
                ):
                    if article_id != _prev_aid:
                        _aid_idx += 1
                        log.info(
                            "glirel_article_start",
                            article=f"{_aid_idx}/{_n_arts}",
                            article_id=article_id,
                            entities_in_chunk=len(ents),
                        )
                        _prev_aid = article_id
                    scored = glirel.score_chunk(
                        chunk_text, ents, RELATION_LABELS, self.glirel_threshold
                    )
                    for item in scored:
                        _add_relation(
                            per_article_relations[article_id],
                            item["head"], item["relation"], item["tail"], item["score"],
                        )
            else:
                with ThreadPoolExecutor(max_workers=n_workers) as pool:
                    futures = {
                        pool.submit(
                            glirel.score_chunk, chunk_text, ents,
                            RELATION_LABELS, self.glirel_threshold,
                        ): article_id
                        for article_id, chunk_text, ents in work_items
                    }
                    for fut in _progress(
                        as_completed(futures), total=len(futures), desc="glirel_score"
                    ):
                        article_id = futures[fut]
                        try:
                            scored = fut.result()
                        except Exception as e:
                            log.warning(
                                "glirel_chunk_failed",
                                article_id=article_id, error=str(e),
                            )
                            continue
                        for item in scored:
                            _add_relation(
                                per_article_relations[article_id],
                                item["head"], item["relation"], item["tail"], item["score"],
                            )
        finally:
            torch.set_num_threads(prev_threads)

        # ── Convert to flat list ─────────────────────────────────────────────
        for article in articles:
            article_relations = per_article_relations.get(article.id, {})
            for (head, relation, tail), score in article_relations.items():
                all_triples.append({
                    "head":       head,
                    "relation":   relation,
                    "tail":       tail,
                    "score":      round(score, 4),
                    "article_id": article.id,
                })
            if article_relations:
                log.info(
                    "re_article_triples",
                    article_id=article.id,
                    triples=len(article_relations),
                )
                for i, ((head, relation, tail), score) in enumerate(
                    sorted(article_relations.items(), key=lambda x: -x[1])[:3]
                ):
                    log.info(
                        "re_sample_triple",
                        n=f"{i+1}/top3",
                        triple=f"{head} --[{relation}]--> {tail}",
                        score=round(score, 3),
                    )

        log.info(
            "relation_extraction_complete",
            backend="glirel",
            articles=len(articles),
            chunks_scored=len(work_items),
            triples=len(all_triples),
            elapsed_s=round(time.time() - re_t0, 1),
        )
        return all_triples

    # ------------------------------------------------------------------
    # Internal GLiNER NER
    # ------------------------------------------------------------------

    def _extract_gliner_batch(
        self, articles: List[ArticleModel]
    ) -> List[List[EntityMention]]:
        model  = self._load_gliner()
        labels = self.get_gliner_labels()
        results: List[List[EntityMention]] = []

        for article in articles:
            text = article.content or ""
            try:
                entities = model.predict_entities(
                    text, labels, threshold=self.ner_threshold
                )
                mentions = [
                    EntityMention(
                        text=ent["text"],
                        entity_type=ent["label"],
                        confidence=ent.get("score", 0.5),
                        article_id=article.id,
                        span_start=ent["start"],
                        span_end=ent["end"],
                    )
                    for ent in entities
                ]
                results.append(mentions)
            except Exception as e:
                log.warning("gliner_article_failed", article_id=article.id, error=str(e))
                results.append([])

        return results

    # ------------------------------------------------------------------
    # Text chunker
    # ------------------------------------------------------------------

    @staticmethod
    def _chunk_text(text: str, max_words: int = MAX_WORDS) -> List[str]:
        HEADER_RE = re.compile(r"^=+.*=+$")
        paras = [
            ln.strip()
            for ln in text.split("\n")
            if ln.strip() and not HEADER_RE.match(ln.strip())
        ]
        chunks: List[str] = []
        cur: List[str]    = []
        n = 0

        def flush():
            nonlocal cur, n
            if cur:
                chunks.append(" ".join(cur))
                cur, n = [], 0

        for p in paras:
            words = p.split()
            w     = len(words)

            if w > max_words:
                flush()
                for i in range(0, w, max_words):
                    chunks.append(" ".join(words[i:i + max_words]))
                continue

            if n + w > max_words and cur:
                flush()
            cur.append(p)
            n += w

        flush()
        return chunks