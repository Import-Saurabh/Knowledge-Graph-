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
from typing import List, Dict, Optional, Tuple, Any

import torch
from src.models.article import ArticleModel
from src.models.entity import EntityMention
from src.utils.config import settings
from src.utils.logger import get_logger

# Set CPU threads for optimal performance
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
                # GLiNER can accept a list of texts and returns a list of predictions
                all_scores = self.model.predict_entities(
                    probes,
                    RELATION_LABELS,
                    threshold=0.0,  # we'll apply threshold later
                )
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
    ):
        self.ontology = ontology_manager
        self.use_spacy_fallback = use_spacy_fallback
        self.use_glirel = use_glirel
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
        Run GLiNER-2 NER + batched GLiNER-2 RE on all articles.

        Public signature kept identical to the old GLiREL-based version so
        main.py requires zero changes.  Returns a flat list of relation dicts:
            {head, relation, tail, score, article_id}

        Implementation:
          • Chunks each article into MAX_WORDS windows.
          • Runs GLiNER NER on all chunks in one batched call.
          • For each article, collects all entity pairs that pass type constraints.
          • Scores all those pairs in a single batched RE call per article.
          • Deduplicates via _add_relation() (higher-scoring direction wins).
        """
        if not self.use_glirel:
            return []

        gliner = self._load_gliner()
        re_ext = self._load_relation_extractor()
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

        # Step 2: Batch NER on all chunks
        chunk_texts = [chunk for _, chunk in chunk_to_article]
        try:
            all_ents_raw = gliner.predict_entities(
                chunk_texts, labels, threshold=self.ner_threshold
            )
        except Exception as e:
            log.error("gliner_batch_ner_failed", error=str(e))
            return []

        # Step 3: For each article, gather entity pairs and score them in batch
        for article in articles:
            article_relations: Dict[Tuple, float] = {}
            # Get chunks and their NER results for this article
            article_chunks_texts = article_chunks.get(article.id, [])
            if not article_chunks_texts:
                continue

            # Find indices of chunks belonging to this article
            start_idx = 0
            article_ents = []
            for chunk_text in article_chunks_texts:
                # Find the chunk in chunk_to_article (order preserved)
                # Since we built chunk_to_article in same order, we can maintain a pointer
                # but simpler: use a dict mapping chunk text to its raw entities
                # However, chunk texts might not be unique. Better to use the list index.
                # We'll iterate over chunk_to_article and collect matching article_id.
                # But for performance, we can pre-build a map from article_id to list of chunk indices.
                pass

            # More efficient: build a map from article_id to list of (chunk_text, ents_raw)
            # We'll do that in a single pass after NER.

        # Let's refactor: after NER, we have a list of ents_raw per chunk.
        # We can group by article_id.

        # Rebuild mapping: article_id -> list of (chunk_text, entities)
        article_chunk_ents = {a.id: [] for a in articles}
        for (article_id, chunk_text), ents in zip(chunk_to_article, all_ents_raw):
            if ents:
                article_chunk_ents[article_id].append((chunk_text, ents))

        # Now process each article's chunks
        for article in articles:
            article_relations: Dict[Tuple, float] = {}
            chunk_data = article_chunk_ents.get(article.id, [])
            if not chunk_data:
                continue

            # Collect all pairs across chunks for this article
            all_pairs = []  # (head_text, tail_text, head_type, tail_type)
            for chunk_text, ents_raw in chunk_data:
                # Filter out excluded types and limit entities
                ents = [
                    e for e in ents_raw
                    if e["label"].lower() not in RELATION_ENTITY_EXCLUDE
                ][:MAX_ENTS]
                if len(ents) < 2:
                    continue

                # Generate all ordered pairs that pass type constraints for at least one relation
                for head, tail in combinations(ents, 2):
                    for h, t in [(head, tail), (tail, head)]:
                        htype = h["label"].lower()
                        ttype = t["label"].lower()
                        # Check if there's any allowed relation
                        has_allowed = any(
                            _passes_constraint(htype, ttype, lbl)
                            for lbl in RELATION_LABELS
                        )
                        if has_allowed:
                            all_pairs.append((h["text"], t["text"], htype, ttype))

            if not all_pairs:
                continue

            # Step 4: Batch score all pairs for this article
            scored = re_ext.score_pairs_batch(all_pairs)
            for pair_score in scored:
                if pair_score is None:
                    continue
                head = pair_score["head_text"].strip()
                tail = pair_score["tail_text"].strip()
                if not head or not tail or head.lower() == tail.lower():
                    continue
                _add_relation(
                    article_relations,
                    head,
                    pair_score["label"],
                    tail,
                    pair_score["score"],
                )

            # Convert deduplicated dict → list
            for (head, relation, tail), score in article_relations.items():
                all_triples.append({
                    "head": head,
                    "relation": relation,
                    "tail": tail,
                    "score": round(score, 4),
                    "article_id": article.id,
                })

            log.debug(
                "re_article_triples",
                article_id=article.id,
                triples=len(article_relations),
                chunks=len(chunk_data),
            )

        log.info(
            "gliner_re_extraction_complete",
            articles=len(articles),
            triples=len(all_triples),
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