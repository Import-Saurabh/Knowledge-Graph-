"""
src/entities/entity_extractor.py  — GLiREL-Large integration
Ported the zero-shot relation extraction logic from extract_kg.py into a
reusable EntityExtractor method so the pipeline can produce GLiREL triples
alongside (or instead of) LLM triples.

Key design choices (same as extract_kg.py):
  • RELATION_ENTITY_EXCLUDE  — dates/money are filtered before GLiREL to kill
                               "2024 --sanctioned by--> Iran" junk.
  • CONSTRAINTS dict         — type-aware post-filter on (head_type, tail_type)
                               so "killed" only fires when tail is a person, etc.
  • add_relation()           — keeps only the higher-scoring direction for each
                               (head, label, tail) unordered pair.
  • MAX_ENTS / MAX_WORDS     — same chunk-size guardrails as extract_kg.py.
  • extract_with_glirel()    — new public method; returns List[dict] of triples
                               with keys: head, relation, tail, score, article_id
"""

import re
import uuid
from typing import List, Dict, Optional, Tuple

from src.models.article import ArticleModel
from src.models.entity import EntityMention
from src.utils.config import settings
from src.utils.logger import get_logger

log = get_logger(__name__)

BATCH_SIZE = 32
CONFIDENCE_THRESHOLD = 0.4

# ---------------------------------------------------------------------------
# GLiREL configuration — mirrored from extract_kg.py
# ---------------------------------------------------------------------------

MAX_ENTS = 40          # cap entities per chunk before feeding GLiREL
MAX_WORDS = 200        # words per chunk

RELATION_ENTITY_EXCLUDE = {"date", "money or economic value"}

TOKEN_RE = re.compile(r"\w+(?:[-_]\w+)*|\S")

# Logical actor / location / thing groups used in CONSTRAINTS
_PER    = {"person"}
_ROLE   = {"job title or role"}
_NATION = {"country", "geopolitical entity"}
_ORG    = {"organization", "government agency", "military unit", "political group"}
_PLACE  = {"country", "city", "location", "geopolitical entity", "facility"}
_ARMS   = {"weapon", "military operation", "vehicle or aircraft"}
_EVENT  = {"event", "military operation"}
_ACTOR  = _PER | _ORG | _NATION
_HITTABLE = _PLACE | _ORG | _PER | _ARMS

GLIREL_RELATION_LABELS = [
    "allied with", "at war with", "attacked", "invaded", "located in",
    "capital of", "part of", "member of", "leader of", "head of state of",
    "headquartered in", "supported by", "funded by", "sanctioned by",
    "negotiated with", "signed agreement with", "killed", "based in",
    "occurred in", "controlled by", "founded by", "spokesperson of",
    "citizen of", "participant in", "targeted", "responded to",
    "threatened", "supplied weapons to", "accused",
]

GLIREL_CONSTRAINTS: Dict[str, Tuple[set, set]] = {
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


def _constrain_glirel(rels, ner):
    """Drop relations whose head/tail type violates the logical rule."""
    span_label = {(n[0], n[1]): n[2] for n in ner}
    out = []
    for r in rels:
        rule = GLIREL_CONSTRAINTS.get(r["label"])
        if rule is None:
            out.append(r)
            continue
        # GLiREL end index is exclusive; subtract 1 to match ner span format
        head_t = span_label.get((r["head_pos"][0], r["head_pos"][1] - 1))
        tail_t = span_label.get((r["tail_pos"][0], r["tail_pos"][1] - 1))
        if head_t in rule[0] and tail_t in rule[1]:
            out.append(r)
    return out


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


# ---------------------------------------------------------------------------
# Main extractor class
# ---------------------------------------------------------------------------

class EntityExtractor:
    def __init__(
        self,
        use_spacy_fallback: bool = False,
        ontology_manager=None,
        use_glirel: bool = True,
        glirel_threshold: float = 0.50,
        ner_threshold: float = CONFIDENCE_THRESHOLD,
    ):
        self.ontology = ontology_manager
        self.use_spacy_fallback = use_spacy_fallback
        self.use_glirel = use_glirel
        self.glirel_threshold = glirel_threshold
        self.ner_threshold = ner_threshold
        self._gliner_model = None
        self._glirel_model = None
        self._spacy_nlp = None

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

    def _load_glirel(self):
        if self._glirel_model is None:
            try:
                from glirel import GLiREL
                try:
                    self._glirel_model = GLiREL.from_pretrained(
                        "jackboyla/glirel-large-v0"
                    )
                except TypeError:
                    # Newer huggingface_hub drops kwargs GLiREL's mixin still requires
                    self._glirel_model = GLiREL._from_pretrained(
                        model_id="jackboyla/glirel-large-v0",
                        revision=None, cache_dir=None, force_download=False,
                        proxies=None, resume_download=False,
                        local_files_only=False, token=None, map_location="cpu",
                    )
                log.info("glirel_model_loaded", model="jackboyla/glirel-large-v0")
            except Exception as e:
                log.error("glirel_load_failed", error=str(e))
                raise
        return self._glirel_model

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
        NEW: Run GLiNER (entities) + GLiREL-Large (relations) on each article.
        Returns a flat list of relation dicts:
            {head, relation, tail, score, article_id, head_type, tail_type}

        Design mirrors extract_kg.py:
          • chunk each article into MAX_WORDS windows
          • filter RELATION_ENTITY_EXCLUDE entity types before GLiREL
          • apply GLIREL_CONSTRAINTS post-filter
          • _add_relation() deduplication (higher-score direction wins)
        """
        if not self.use_glirel:
            return []

        gliner = self._load_gliner()
        glirel = self._load_glirel()
        labels = self.get_gliner_labels()
        all_triples: List[Dict] = []

        for article in articles:
            text = (article.content or "")
            if not text.strip():
                continue

            chunks = self._chunk_text(text)
            article_relations: Dict[Tuple, float] = {}

            for chunk in chunks:
                ents_raw = gliner.predict_entities(
                    chunk, labels, threshold=self.ner_threshold
                )
                toks, spans = _tokenize(chunk)

                # Build NER list for GLiREL, excluding date/money types
                ner, seen = [], set()
                for ent in sorted(ents_raw, key=lambda x: -x["score"]):
                    if ent["label"].lower() in RELATION_ENTITY_EXCLUDE:
                        continue
                    ts = _char_to_token_span(ent["start"], ent["end"], spans)
                    if not ts or ts in seen:
                        continue
                    seen.add(ts)
                    ner.append([ts[0], ts[1], ent["label"], ent["text"]])
                    if len(ner) >= MAX_ENTS:
                        break

                if len(ner) < 2:
                    continue

                try:
                    rels = glirel.predict_relations(
                        toks,
                        GLIREL_RELATION_LABELS,
                        threshold=self.glirel_threshold,
                        ner=ner,
                        top_k=1,
                    )
                    rels = _constrain_glirel(rels, ner)
                except Exception as e:
                    log.warning(
                        "glirel_chunk_failed",
                        article_id=article.id,
                        error=str(e),
                    )
                    continue

                # Build span→label lookup for type metadata
                span_label = {(n[0], n[1]): n[2] for n in ner}

                for r in rels:
                    head = " ".join(r["head_text"]).strip()
                    tail = " ".join(r["tail_text"]).strip()
                    if head.lower() == tail.lower():
                        continue
                    _add_relation(article_relations, head, r["label"], tail, r["score"])

            # Convert deduplicated dict → list of triples
            for (head, relation, tail), score in article_relations.items():
                all_triples.append(
                    {
                        "head": head,
                        "relation": relation,
                        "tail": tail,
                        "score": round(score, 4),
                        "article_id": article.id,
                    }
                )

        log.info(
            "glirel_extraction_complete",
            articles=len(articles),
            triples=len(all_triples),
        )
        return all_triples

    # ------------------------------------------------------------------
    # Internal GLiNER extraction
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
            "PERSON": "person",
            "ORG": "organization",
            "GPE": "country",
            "LOC": "location",
            "EVENT": "event",
            "PRODUCT": "product",
            "WORK_OF_ART": "product",
            "LAW": "law or sanction",
            "NORP": "political group",
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
    # Chunking helper (same logic as extract_kg.py)
    # ------------------------------------------------------------------

    @staticmethod
    def _chunk_text(text: str, max_words: int = MAX_WORDS) -> List[str]:
        HEADER_RE = re.compile(r"^=+.*=+$")
        paras = [
            ln.strip()
            for ln in text.split("\n")
            if ln.strip() and not HEADER_RE.match(ln.strip())
        ]
        chunks, cur, n = [], [], 0
        for p in paras:
            w = len(p.split())
            if n + w > max_words and cur:
                chunks.append(" ".join(cur))
                cur, n = [], 0
            cur.append(p)
            n += w
        if cur:
            chunks.append(" ".join(cur))
        return chunks