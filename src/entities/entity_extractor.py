"""
src/entities/entity_extractor.py

GLiREL REMOVED — replaced with a pure GLiNER-2 relation extractor.

Reasons for removal:
  • GLiREL._from_pretrained() in current pip versions requires 'proxies' and
    'resume_download' kwargs that the library doesn't pass → always crashes.
  • The published GLiREL Hub models (jackboyla/glirel-large-v0,
    relax/relax-large) have mismatched config schemas that break loading.

Replacement — GliNERRelationExtractor:
  • Uses the SAME urchade/gliner_large-v2.1 model already loaded for NER.
  • For each article chunk, extracts entity pairs then asks GLiNER to score
    each (head, relation_label, tail) triple by wrapping the relation as a
    typed span label and checking the model's entity-score for that label.
  • Falls back gracefully: if the relation score is below threshold the
    triple is discarded.
  • Same public interface as the old GLiREL path:
      extract_with_glirel(articles) -> List[dict]   (key name kept for
      compatibility so main.py needs no changes)

All other behaviour (GLiNER NER, spaCy fallback, chunking, CONSTRAINTS,
_add_relation dedup) is unchanged.
"""

import re
import uuid
from itertools import combinations
from typing import List, Dict, Optional, Tuple

from src.models.article import ArticleModel
from src.models.entity import EntityMention
from src.utils.config import settings
from src.utils.logger import get_logger

log = get_logger(__name__)

BATCH_SIZE = 32
CONFIDENCE_THRESHOLD = 0.4

# ---------------------------------------------------------------------------
# GLiNER-2 RE configuration
# ---------------------------------------------------------------------------

MAX_ENTS  = 40    # cap entities per chunk
MAX_WORDS = 200   # words per chunk

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
        return True  # no constraint → always allow
    return (head_type in rule[0]) and (tail_type in rule[1])


# ---------------------------------------------------------------------------
# GLiNER-2 relation extractor (replaces GLiREL)
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
    GLiNER model.
    """

    def __init__(self, gliner_model, threshold: float = 0.45):
        self.model = gliner_model
        self.threshold = threshold

    def predict(
        self,
        text: str,
        entities: List[dict],   # [{text, label, start, end, score}, ...]
    ) -> List[dict]:
        """
        Returns list of relation dicts:
            {head_text, tail_text, head_type, tail_type, label, score}
        """
        # Filter out excluded entity types
        ents = [
            e for e in entities
            if e["label"].lower() not in RELATION_ENTITY_EXCLUDE
        ][:MAX_ENTS]

        if len(ents) < 2:
            return []

        results = []

        for head, tail in combinations(ents, 2):
            # Check both directions: head→tail and tail→head
            for h, t in [(head, tail), (tail, head)]:
                h_type = h["label"].lower()
                t_type = t["label"].lower()

                # Collect candidate relations that pass type constraints
                candidate_labels = [
                    lbl for lbl in RELATION_LABELS
                    if _passes_constraint(h_type, t_type, lbl)
                ]
                if not candidate_labels:
                    continue

                # Build a compact context sentence for scoring
                # Format: "<HEAD> [SEP] <TAIL>" — GLiNER scores this
                # against each relation label as if it were an entity type
                probe = f"{h['text']} [and] {t['text']}"

                try:
                    scored = self.model.predict_entities(
                        probe,
                        candidate_labels,
                        threshold=self.threshold,
                    )
                except Exception:
                    continue

                if not scored:
                    continue

                # Pick the highest-confidence relation
                best = max(scored, key=lambda x: x["score"])
                results.append({
                    "head_text": h["text"],
                    "tail_text": t["text"],
                    "head_type": h_type,
                    "tail_type": t_type,
                    "label":     best["label"],
                    "score":     best["score"],
                })

        return results


# ---------------------------------------------------------------------------
# Main extractor class
# ---------------------------------------------------------------------------

class EntityExtractor:
    def __init__(
        self,
        use_spacy_fallback: bool = False,
        ontology_manager=None,
        use_glirel: bool = True,     # kept for API compatibility; controls RE pass
        glirel_threshold: float = 0.45,
        ner_threshold: float = CONFIDENCE_THRESHOLD,
    ):
        self.ontology = ontology_manager
        self.use_spacy_fallback = use_spacy_fallback
        self.use_glirel = use_glirel          # now controls GLiNER-RE pass
        self.glirel_threshold = glirel_threshold
        self.ner_threshold = ner_threshold
        self._gliner_model = None
        self._re_extractor: Optional[_GliNERRelationExtractor] = None
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

    def _load_relation_extractor(self) -> _GliNERRelationExtractor:
        """Return the GLiNER-2 based RE model (lazy init, reuses NER model)."""
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
        Run GLiNER-2 NER + GLiNER-2 RE on each article.

        Public signature kept identical to the old GLiREL-based version so
        main.py requires zero changes.  Returns a flat list of relation dicts:
            {head, relation, tail, score, article_id}

        Implementation:
          • Chunks each article into MAX_WORDS windows.
          • Runs GLiNER NER on the chunk to get typed entities.
          • Passes entity pairs through _GliNERRelationExtractor which probes
            the same GLiNER model with synthetic prompts to score relation labels.
          • Deduplicates via _add_relation() (higher-scoring direction wins).
        """
        if not self.use_glirel:
            return []

        gliner = self._load_gliner()
        re_ext = self._load_relation_extractor()
        labels = self.get_gliner_labels()
        all_triples: List[Dict] = []

        for article in articles:
            text = (article.content or "").strip()
            if not text:
                continue

            chunks = self._chunk_text(text)
            article_relations: Dict[Tuple, float] = {}
            chunk_count = 0

            for chunk in chunks:
                chunk_count += 1

                # Step 1: NER
                try:
                    ents_raw = gliner.predict_entities(
                        chunk, labels, threshold=self.ner_threshold
                    )
                except Exception as e:
                    log.warning(
                        "gliner_chunk_ner_failed",
                        article_id=article.id, error=str(e),
                    )
                    continue

                if len(ents_raw) < 2:
                    continue

                # Step 2: RE via GLiNER-2 pair scoring
                try:
                    rels = re_ext.predict(chunk, ents_raw)
                except Exception as e:
                    log.warning(
                        "gliner_re_chunk_failed",
                        article_id=article.id, error=str(e),
                    )
                    continue

                for r in rels:
                    head = r["head_text"].strip()
                    tail = r["tail_text"].strip()
                    if not head or not tail or head.lower() == tail.lower():
                        continue
                    _add_relation(
                        article_relations, head, r["label"], tail, r["score"]
                    )

            # Convert deduplicated dict → list
            for (head, relation, tail), score in article_relations.items():
                all_triples.append(
                    {
                        "head":       head,
                        "relation":   relation,
                        "tail":       tail,
                        "score":      round(score, 4),
                        "article_id": article.id,
                    }
                )

            log.debug(
                "re_article_triples",
                article_id=article.id,
                triples=len(article_relations),
                chunks=chunk_count,
            )

        log.info(
            "gliner_re_extraction_complete",
            articles=len(articles),
            triples=len(all_triples),
        )
        return all_triples

    # ------------------------------------------------------------------
    # Internal GLiNER NER extraction
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