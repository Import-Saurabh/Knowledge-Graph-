"""
src/entities/entity_extractor.py

GLiREL RESTORED (real model), with a GLiNER-based fallback.

History
  • GLiREL was previously removed because `GLiREL.from_pretrained()` crashed
    with `TypeError: ..._from_pretrained() missing 2 required keyword-only
    arguments: 'proxies' and 'resume_download'`. That's a version mismatch
    between `glirel` and `huggingface_hub` (newer hub releases stopped
    forwarding those kwargs), not a problem with GLiREL itself — fix it with
    `pip install "huggingface_hub==0.24.6" glirel --break-system-packages`.
  • The GLiNER-as-relation-classifier replacement that took GLiREL's place
    also had a hard bug: every call site passed a *list* of texts into
    `GLiNER.predict_entities()`, which only accepts a single string. The
    exception was swallowed, so relation extraction always silently
    returned 0 triples. Fixed via `_batch_predict()` below, which uses the
    real `batch_predict_entities()` API.

Current design
  • `_load_glirel()` tries to load real GLiREL (`jackboyla/glirel_base`) on
    first use. If that succeeds, `_RealGlirelExtractor` does genuine
    zero-shot relation extraction using the SAME GLiNER entities already
    extracted for NER (converted from character spans to token spans via
    `_tokenize`/`_char_to_token_span`).
  • If GLiREL can't load in this environment, `extract_with_glirel()` falls
    back to `_GliNERRelationExtractor`, the GLiNER-as-relation-classifier
    workaround — now with its batching bug fixed, but still a lower-recall
    approximation since it isn't a trained relation-classification model.
  • Same public interface either way:
      extract_with_glirel(articles) -> List[dict]   (key name kept for
      compatibility so main.py needs no changes)

All other behaviour (GLiNER NER, spaCy fallback, chunking, CONSTRAINTS,
_add_relation dedup) is unchanged.
"""

import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from typing import List, Dict, Optional, Tuple, Any

import torch
from src.models.article import ArticleModel
from src.models.entity import EntityMention
from src.utils.config import settings
from src.utils.logger import get_logger

# Set CPU threads for optimal performance. This is the baseline used for
# single-threaded calls (the batched GLiNER NER pass). It's temporarily
# reduced inside extract_with_glirel() while GLiREL scoring fans out across
# worker threads, so we don't oversubscribe the CPU.
torch.set_num_threads(8)  # adjust based on your CPU cores

log = get_logger(__name__)

BATCH_SIZE = 32
CONFIDENCE_THRESHOLD = 0.4

# ---------------------------------------------------------------------------
# GLiNER-2 RE configuration
# ---------------------------------------------------------------------------

MAX_ENTS = 40          # cap entities per chunk
MAX_WORDS = 400        # increased from 200 to reduce chunk count
RELATION_ENTITY_EXCLUDE = {"date", "money or economic value"}

TOKEN_RE = re.compile(r"\w+(?:[-_]\w+)*|\S")

# Logical actor / location / thing groups used in CONSTRAINTS
_PER = {"person"}
_ROLE = {"job title or role"}
_NATION = {"country", "geopolitical entity"}
_ORG = {"organization", "government agency", "military unit", "political group"}
_PLACE = {"country", "city", "location", "geopolitical entity", "facility"}
_ARMS = {"weapon", "military operation", "vehicle or aircraft"}
_EVENT = {"event", "military operation"}
_ACTOR = _PER | _ORG | _NATION
_HITTABLE = _PLACE | _ORG | _PER | _ARMS

# Relation labels we want to detect — same list as before
RELATION_LABELS = [
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


def _add_relation(relations: dict, head: str, label: str, tail: str, score: float):
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
    BUG FIX: `GLiNER.predict_entities()` only accepts a SINGLE string as its
    first argument. Every call-site in this file used to pass a *list* of
    texts straight into `predict_entities()` (e.g. `predict_entities(probes,
    RELATION_LABELS, ...)` and `predict_entities(chunk_texts, labels, ...)`).
    That raises inside the model on every call, which was silently caught by
    a wrapping try/except and turned into an empty result — the root cause
    of GLiREL/relation-extraction always returning 0 triples.

    GLiNER's real batch API is `batch_predict_entities(texts, labels,
    threshold=...)`. We prefer it when present and fall back to looping the
    single-text API for older `gliner` versions that don't expose it.
    """
    if hasattr(model, "batch_predict_entities"):
        return model.batch_predict_entities(texts, labels, threshold=threshold)
    return [model.predict_entities(t, labels, threshold=threshold) for t in texts]


def _progress(iterable, total: Optional[int] = None, desc: str = ""):
    """
    Iterate over `iterable`, showing a tqdm progress bar if tqdm is
    installed, otherwise falling back to periodic structured log lines
    (~20 updates over the run, with elapsed/ETA) so progress is visible
    either way. This is what was missing before — the GLiREL scoring loop
    used to run silently for however long it took with no feedback at all.
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
# GLiNER-2 relation extractor (replaces GLiREL) — now with batched scoring
# ---------------------------------------------------------------------------

class _GliNERRelationExtractor:
    """
    Uses GLiNER-2 to score entity-pair relations.

    Strategy:
      1. Extract named entities from a text chunk with GLiNER.
      2. For each ordered entity pair (head, tail) whose types pass the
         constraint filter, build a synthetic prompt:
             "<head_text> [RELATION] <tail_text>"
         and ask GLiNER to score it against each relation label.
      3. Keep the highest-scoring label above threshold.

    This avoids any dependency on GLiREL while reusing the already-loaded
    GLiNER model.  Scoring is batched across many pairs for efficiency.
    """

    def __init__(self, gliner_model, threshold: float = 0.45):
        self.model = gliner_model
        self.threshold = threshold

    def score_pairs_batch(
        self,
        pairs: List[Tuple[str, str, str, str]],  # (head_text, tail_text, head_type, tail_type)
        batch_size: int = 256,
    ) -> List[Optional[Dict[str, Any]]]:
        """
        Score a list of entity pairs against all relation labels in one batched GLiNER call.

        Returns a list of dicts (or None) with keys:
            head_text, tail_text, head_type, tail_type, label, score
        """
        if not pairs:
            return []

        results = [None] * len(pairs)
        # Batch to avoid OOM
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start:start+batch_size]
            probes = [f"{h} [and] {t}" for h, t, _, _ in batch]
            try:
                # BUG FIX: was `self.model.predict_entities(probes, ...)` —
                # predict_entities() only takes ONE string; passing a list
                # raised on every call and was swallowed below, so this path
                # always returned zero relations. Use the batch-safe helper.
                all_scores = _batch_predict(self.model, probes, RELATION_LABELS, 0.0)
            except Exception as e:
                log.warning("gliner_batch_re_failed", error=str(e))
                continue

            for idx_in_batch, (probe, scores_for_probe, pair) in enumerate(
                zip(probes, all_scores, batch)
            ):
                h, t, htype, ttype = pair
                # Find allowed relations
                allowed = [
                    lbl for lbl in RELATION_LABELS
                    if _passes_constraint(htype, ttype, lbl)
                ]
                best_label = None
                best_score = self.threshold
                for item in scores_for_probe:
                    if item["label"] in allowed and item["score"] > best_score:
                        best_score = item["score"]
                        best_label = item["label"]
                if best_label is not None:
                    results[start + idx_in_batch] = {
                        "head_text": h,
                        "tail_text": t,
                        "head_type": htype,
                        "tail_type": ttype,
                        "label": best_label,
                        "score": best_score,
                    }
        return results


# ---------------------------------------------------------------------------
# Real GLiREL relation extractor (jackboyla/GLiREL)
# ---------------------------------------------------------------------------
#
# This is the GENUINE GLiREL model, not the GLiNER-as-relation-classifier
# workaround above. It was previously removed because `GLiREL.from_pretrained()`
# crashed with:
#
#   TypeError: GLiREL._from_pretrained() missing 2 required keyword-only
#   arguments: 'proxies' and 'resume_download'
#
# That is a version-mismatch bug between `glirel` and `huggingface_hub`, NOT
# a problem with GLiREL itself: newer `huggingface_hub` releases dropped
# `proxies`/`resume_download` from the kwargs they forward through
# `ModelHubMixin.from_pretrained()`, but `glirel`'s vendored hub-mixin code
# still declares them as required. Fix the environment, don't avoid the
# library:
#
#       pip install "huggingface_hub==0.24.6" glirel --break-system-packages
#
# (0.23.x/0.25.x also work — anything before the kwarg was dropped). We still
# guard the import/load with try/except below so the pipeline degrades to
# the GLiNER fallback instead of crashing if that pin isn't in place yet.
#
# GLiREL needs token-index entity spans (`[start_tok, end_tok_inclusive,
# type, text]`) rather than character offsets. We reuse the already-loaded
# GLiNER entities (richer domain types than a generic spaCy NER pass would
# give us) and convert their character spans to token spans with the
# `_tokenize` / `_char_to_token_span` helpers defined above.

class _RealGlirelExtractor:
    """Thin wrapper around `glirel.GLiREL` (Boylan et al., 2025)."""

    # "jackboyla/glirel_base" is the lighter model — appropriate for CPU-only
    # machines. Swap in "jackboyla/glirel-large-v0" for higher accuracy if
    # you have the RAM/CPU budget to spare.
    MODEL_NAME = "jackboyla/glirel-large-v0"

    def __init__(self, model_name: Optional[str] = None):
        import inspect
        from glirel import GLiREL  # raises ImportError if `glirel` isn't installed

        # ------------------------------------------------------------------ #
        # FIX: huggingface_hub ≥ 0.25 stopped forwarding `proxies` and       #
        # `resume_download` through ModelHubMixin.from_pretrained(), but      #
        # glirel's _from_pretrained() still declares them as *required*       #
        # keyword-only args → TypeError every time.                           #
        #                                                                     #
        # We inspect the live signature and, if those params are present and  #
        # have no default, wrap _from_pretrained with a shim that injects     #
        # safe defaults so the call succeeds with any hub version.            #
        # ------------------------------------------------------------------ #
        try:
            raw = GLiREL._from_pretrained
            fn = raw.__func__ if hasattr(raw, "__func__") else raw
            sig = inspect.signature(fn)
            params = sig.parameters
            needs_patch = any(
                name in params and params[name].default is inspect.Parameter.empty
                for name in ("proxies", "resume_download")
            )
            if needs_patch:
                _orig_fn = fn  # capture

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
            pass  # if inspection itself fails, just let from_pretrained try and error naturally

        self.model = GLiREL.from_pretrained(model_name or self.MODEL_NAME)

    def score_chunk(
        self,
        chunk_text: str,
        gliner_ents: List[dict],
        relation_labels: List[str],
        threshold: float,
    ) -> List[Dict[str, Any]]:
        """
        gliner_ents: GLiNER entity dicts with character-offset 'start'/'end'.
        Returns a list of {head, tail, relation, score} dicts.
        """
        if len(gliner_ents) < 2:
            return []

        tokens, spans = _tokenize(chunk_text)
        if not tokens:
            return []

        ner = []
        for ent in gliner_ents:
            tok_span = _char_to_token_span(ent["start"], ent["end"], spans)
            if tok_span is None:
                continue
            # GLiREL's end index is INCLUSIVE — unlike spaCy/GLiNER's
            # exclusive convention — so we keep tok_span as-is (it's already
            # the inclusive (first_token_idx, last_token_idx) pair).
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
            head = " ".join(r.get("head_text") or []).strip()
            tail = " ".join(r.get("tail_text") or []).strip()
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
    def __init__(
        self,
        use_spacy_fallback: bool = False,
        ontology_manager=None,
        use_glirel: bool = True,
        glirel_threshold: float = 0.45,
        ner_threshold: float = CONFIDENCE_THRESHOLD,
        glirel_workers: int = 4,
    ):
        self.ontology = ontology_manager
        self.use_spacy_fallback = use_spacy_fallback
        self.use_glirel = use_glirel
        self.glirel_threshold = glirel_threshold
        self.ner_threshold = ner_threshold
        # Worker threads for the GLiREL scoring fan-out (see
        # extract_with_glirel). Your i5-1240P is 4 P-cores + 8 E-cores —
        # 4 workers x ~2-3 intra-op threads each is a sane starting point.
        # Set glirel_workers=1 to fall back to the original fully
        # sequential behavior if you ever hit weirdness under concurrency.
        self.glirel_workers = max(1, glirel_workers)
        self._gliner_model = None
        self._re_extractor: Optional[_GliNERRelationExtractor] = None
        self._spacy_nlp = None
        self._glirel_model: Optional[_RealGlirelExtractor] = None
        self._glirel_unavailable = False

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

    def _load_relation_extractor(self) -> _GliNERRelationExtractor:
        if self._re_extractor is None:
            model = self._load_gliner()
            self._re_extractor = _GliNERRelationExtractor(
                gliner_model=model,
                threshold=self.glirel_threshold,
            )
            log.info(
                "gliner_re_loaded",
                model="urchade/gliner_large-v2.1",
                threshold=self.glirel_threshold,
            )
        return self._re_extractor

    def _load_glirel(self) -> Optional[_RealGlirelExtractor]:
        """
        Lazily load the real GLiREL model. Returns None (logging once) if it
        can't load, so callers fall back to the GLiNER-based pseudo-RE path
        instead of crashing the whole pipeline.
        """
        if self._glirel_unavailable:
            return None
        if self._glirel_model is not None:
            return self._glirel_model

        # 1) Try local cache only, zero network calls. GLiREL.from_pretrained()
        # normally does a HEAD request to huggingface.co even when the model
        # is already fully cached — a transient DNS/Wi-Fi blip during that
        # check kills the whole load even though the files are on disk.
        # HF_HUB_OFFLINE=1 skips that check and reads straight from cache.
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
            pass  # not cached (or cache incomplete) — fall through to a real load
        finally:
            if prev_offline is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = prev_offline

        # 2) Not cached — do a real network load, retrying a few times in
        # case it's just a transient blip (a DNS resolution failure mid-run
        # usually means a brief connectivity drop, not a permanent problem).
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
        is_network_error = any(
            s in err_str for s in (
                "getaddrinfo failed", "NameResolutionError", "ConnectionError",
                "Max retries exceeded", "NewConnectionError", "ConnectTimeout",
                "ProxyError",
            )
        )
        if is_network_error:
            hint = (
                "Real GLiREL failed to load because of a network/DNS error "
                "reaching huggingface.co — NOT a version mismatch. "
                "'jackboyla/glirel_base' isn't fully cached locally yet, so "
                "it needs a successful download at least once. Check your "
                "connection and rerun; after one successful download it'll "
                "load from local cache and won't need network again. "
                "Falling back to the degraded GLiNER-based relation "
                "classifier for this run."
            )
        else:
            hint = (
                "Real GLiREL failed to load. The proxies/resume_download "
                "patch is already applied in code, so this is likely a "
                "different error (model file corrupt/missing, OOM, or "
                "glirel not installed). Try: pip install glirel "
                "--break-system-packages  and rerun. If it worked once "
                "before, delete the cached model dir "
                "(~/.cache/huggingface/hub/models--jackboyla--glirel_base) "
                "and let it re-download. Falling back to the degraded "
                "GLiNER-based relation classifier for this run."
            )
        log.warning("glirel_load_failed_using_fallback", error=err_str, hint=hint)
        return None

    def _load_spacy(self):
        if self._spacy_nlp is None:
            try:
                import spacy
                self._spacy_nlp = spacy.load("en_core_web_sm")
                log.info("spacy_model_loaded")
            except Exception as e:
                log.error("spacy_load_failed", error=str(e))
                raise
        return self._spacy_nlp

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
    # Public extraction API
    # ------------------------------------------------------------------

    def extract_batch(
        self, articles: List[ArticleModel]
    ) -> List[List[EntityMention]]:
        if self.use_spacy_fallback:
            return [self._extract_spacy(a) for a in articles]
        try:
            return self._extract_gliner_batch(articles)
        except Exception as e:
            log.warning("gliner_extraction_failed", error=str(e))
            if self.use_spacy_fallback:
                return [self._extract_spacy(a) for a in articles]
            raise

    def extract_single(self, article: ArticleModel) -> List[EntityMention]:
        if self.use_spacy_fallback:
            return self._extract_spacy(article)
        try:
            return self._extract_gliner_batch([article])[0]
        except Exception as e:
            log.warning("gliner_single_failed", error=str(e))
            if self.use_spacy_fallback:
                return self._extract_spacy(article)
            return []

    def extract_with_glirel(
        self, articles: List[ArticleModel]
    ) -> List[Dict]:
        """
        Run GLiNER NER, then relation extraction over the discovered entity
        pairs. Public signature kept identical to the original GLiREL-based
        version so main.py requires zero changes. Returns a flat list of
        relation dicts: {head, relation, tail, score, article_id}

        Relation-extraction backend (tried in order):
          1. Real GLiREL (`jackboyla/glirel_base`) — genuine zero-shot RE.
             Uses the GLiNER entities already extracted as `ner` spans, so
             it gets this domain's fine-grained types for free.
          2. GLiNER-as-relation-classifier fallback — used only if GLiREL
             itself can't load in this environment (see `_load_glirel`).

        Implementation:
          • Chunks each article into MAX_WORDS windows.
          • Runs GLiNER NER on all chunks in one true batched call.
          • For each article, runs the chosen RE backend per chunk/pair.
          • Deduplicates via _add_relation() (higher-scoring direction wins).
        """
        if not self.use_glirel:
            return []

        gliner = self._load_gliner()
        labels = self.get_gliner_labels()
        all_triples: List[Dict] = []

        # Step 1: Build all chunks and their article mapping
        chunk_to_article = []          # list of (article_id, chunk_text)
        article_chunks = {a.id: [] for a in articles}
        for article in articles:
            text = (article.content or "").strip()
            if not text:
                continue
            chunks = self._chunk_text(text)
            for chunk in chunks:
                article_chunks[article.id].append(chunk)
                chunk_to_article.append((article.id, chunk))

        if not chunk_to_article:
            return []

        # Step 2: Batch NER on all chunks (BUG FIX: use _batch_predict, not
        # predict_entities(list_of_texts, ...) which always raised).
        # Sub-batched (instead of one call holding every chunk in the run)
        # so you get progress feedback and a single bad sub-batch doesn't
        # abort the whole stage.
        chunk_texts = [chunk for _, chunk in chunk_to_article]
        all_ents_raw: List[List[dict]] = []
        n_sub_batches = (len(chunk_texts) + BATCH_SIZE - 1) // BATCH_SIZE
        ner_t0 = time.time()
        for start in _progress(
            range(0, len(chunk_texts), BATCH_SIZE), total=n_sub_batches, desc="gliner_ner"
        ):
            sub_batch = chunk_texts[start:start + BATCH_SIZE]
            try:
                all_ents_raw.extend(_batch_predict(gliner, sub_batch, labels, self.ner_threshold))
            except Exception as e:
                log.error("gliner_batch_ner_failed", batch_start=start, error=str(e))
                all_ents_raw.extend([[] for _ in sub_batch])
        log.info("gliner_ner_complete", chunks=len(chunk_texts), elapsed_s=round(time.time() - ner_t0, 1))

        # Rebuild mapping: article_id -> list of (chunk_text, entities)
        article_chunk_ents = {a.id: [] for a in articles}
        for (article_id, chunk_text), ents in zip(chunk_to_article, all_ents_raw):
            if ents:
                article_chunk_ents[article_id].append((chunk_text, ents))

        # Step 3: Pick a relation-extraction backend once for the whole run.
        _chunks_with_ents = sum(len(v) for v in article_chunk_ents.values())
        log.info(
            "glirel_loading",
            articles=len(articles),
            chunks_with_ents=_chunks_with_ents,
            hint="Loading GLiREL model — ~10-30s on first call, instant from cache after that...",
        )
        glirel = self._load_glirel()
        re_ext = None if glirel is not None else self._load_relation_extractor()
        backend = "glirel" if glirel is not None else "gliner_fallback"
        log.info("glirel_backend_selected", backend=backend)

        # ------------------------------------------------------------------
        # Step 4: Score relations
        #
        # THE BUG: this used to call glirel.score_chunk() (one full CPU
        # transformer forward pass) once per chunk, strictly sequentially,
        # inside a plain `for article in articles` loop — no batching, no
        # parallelism, and nothing printed until the whole stage finished.
        # With hundreds/thousands of chunks and no GPU, that's "hours with
        # no idea how far along it is."
        #
        # THE FIX: flatten article+chunk into one global work list (one
        # progress bar instead of silence), and fan the GLiREL calls out
        # across a small thread pool — PyTorch's CPU kernels release the
        # GIL during the actual forward pass, so concurrent threads really
        # do use otherwise-idle cores instead of just serializing on Python.
        # ------------------------------------------------------------------

        work_items: List[Tuple[str, str, List[dict]]] = []  # (article_id, chunk_text, filtered_ents)
        for article in articles:
            for chunk_text, ents_raw in article_chunk_ents.get(article.id, []):
                ents = [
                    e for e in ents_raw
                    if e["label"].lower() not in RELATION_ENTITY_EXCLUDE
                ][:MAX_ENTS]
                if len(ents) >= 2:
                    work_items.append((article.id, chunk_text, ents))

        log.info(
            "glirel_scoring_start",
            backend=backend,
            total_chunks=len(work_items),
            total_articles=len([a for a in articles if article_chunk_ents.get(a.id)]),
        )
        per_article_relations: Dict[str, Dict[Tuple, float]] = {a.id: {} for a in articles}
        re_t0 = time.time()

        if glirel is not None:
            # ---- Real GLiREL path, fanned out across worker threads ----
            n_workers = self.glirel_workers
            prev_threads = torch.get_num_threads()
            # Avoid oversubscribing the CPU: split intra-op threads across
            # the worker threads we're about to launch.
            torch.set_num_threads(max(1, prev_threads // n_workers))
            try:
                if n_workers == 1:
                    _prev_article_id = None
                    _article_idx = 0
                    _article_ids_ordered = list(dict.fromkeys(aid for aid, _, _ in work_items))
                    _n_articles = len(_article_ids_ordered)
                    for article_id, chunk_text, ents in _progress(
                        work_items, total=len(work_items), desc="glirel_score"
                    ):
                        if article_id != _prev_article_id:
                            _article_idx += 1
                            log.info(
                                "glirel_article_start",
                                article=f"{_article_idx}/{_n_articles}",
                                article_id=article_id,
                                entities_in_chunk=len(ents),
                            )
                            _prev_article_id = article_id
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
                                log.warning("glirel_chunk_failed", article_id=article_id, error=str(e))
                                continue
                            for item in scored:
                                _add_relation(
                                    per_article_relations[article_id],
                                    item["head"], item["relation"], item["tail"], item["score"],
                                )
            finally:
                torch.set_num_threads(prev_threads)
        else:
            # ---- Fallback: GLiNER-as-relation-classifier ----
            # Flatten pairs across ALL articles into one batch instead of
            # one score_pairs_batch() call per article — fewer, bigger
            # batches mean less per-call overhead.
            flat_pairs = []        # (head_text, tail_text, head_type, tail_type)
            pair_article_ids = []  # which article each pair came from
            for article_id, chunk_text, ents in work_items:
                for head, tail in combinations(ents, 2):
                    for h, t in [(head, tail), (tail, head)]:
                        htype = h["label"].lower()
                        ttype = t["label"].lower()
                        if any(_passes_constraint(htype, ttype, lbl) for lbl in RELATION_LABELS):
                            flat_pairs.append((h["text"], t["text"], htype, ttype))
                            pair_article_ids.append(article_id)

            if flat_pairs:
                scored = re_ext.score_pairs_batch(flat_pairs)
                for article_id, pair_score in _progress(
                    zip(pair_article_ids, scored), total=len(scored), desc="gliner_re_score"
                ):
                    if pair_score is None:
                        continue
                    head = pair_score["head_text"].strip()
                    tail = pair_score["tail_text"].strip()
                    if not head or not tail or head.lower() == tail.lower():
                        continue
                    _add_relation(
                        per_article_relations[article_id],
                        head, pair_score["label"], tail, pair_score["score"],
                    )

        # Convert deduplicated per-article dicts → flat list
        for article in articles:
            article_relations = per_article_relations.get(article.id, {})
            for (head, relation, tail), score in article_relations.items():
                all_triples.append({
                    "head": head,
                    "relation": relation,
                    "tail": tail,
                    "score": round(score, 4),
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
            backend=backend,
            articles=len(articles),
            chunks_scored=len(work_items),
            triples=len(all_triples),
            elapsed_s=round(time.time() - re_t0, 1),
        )
        return all_triples

    # ------------------------------------------------------------------
    # Internal GLiNER NER extraction (unchanged but used for batch)
    # ------------------------------------------------------------------

    def _extract_gliner_batch(
        self, articles: List[ArticleModel]
    ) -> List[List[EntityMention]]:
        model = self._load_gliner()
        labels = self.get_gliner_labels()
        results = []

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
                log.warning(
                    "gliner_article_failed", article_id=article.id, error=str(e)
                )
                results.append([])

        return results

    def _extract_spacy(self, article: ArticleModel) -> List[EntityMention]:
        nlp = self._load_spacy()
        doc = nlp(article.content or "")
        type_map = {
            "PERSON":     "person",
            "ORG":        "organization",
            "GPE":        "country",
            "LOC":        "location",
            "EVENT":      "event",
            "PRODUCT":    "product",
            "WORK_OF_ART":"product",
            "LAW":        "law or sanction",
            "NORP":       "political group",
        }
        return [
            EntityMention(
                text=ent.text,
                entity_type=type_map.get(ent.label_, "Unknown"),
                confidence=0.7,
                article_id=article.id,
                span_start=ent.start_char,
                span_end=ent.end_char,
            )
            for ent in doc.ents
        ]

    # ------------------------------------------------------------------
    # Chunking helper
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
        cur: List[str] = []
        n = 0

        def flush():
            nonlocal cur, n
            if cur:
                chunks.append(" ".join(cur))
                cur, n = [], 0

        for p in paras:
            words = p.split()
            w = len(words)

            if w > max_words:
                # BUG FIX: a paragraph longer than max_words used to be
                # appended whole regardless of size (the old `n + w >
                # max_words and cur` check is False when `cur` is empty),
                # producing oversized chunks that GLiNER/GLiREL silently
                # truncate downstream — losing entities/relations past
                # the cutoff. Split it into max_words-sized pieces instead.
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