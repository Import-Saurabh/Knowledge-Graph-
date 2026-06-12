import re
import uuid
from typing import List
from src.models.article import ArticleModel
from src.models.entity import EntityMention
from src.utils.config import settings
from src.utils.logger import get_logger

log = get_logger(__name__)

BATCH_SIZE = 32
CONFIDENCE_THRESHOLD = 0.4

class EntityExtractor:
    def __init__(self, use_spacy_fallback: bool = False, ontology_manager=None):
        self.ontology = ontology_manager
        self.use_spacy_fallback = use_spacy_fallback
        self._gliner_model = None
        self._spacy_nlp = None

    def _load_gliner(self):
        if self._gliner_model is None:
            try:
                from gliner import GLiNER
                self._gliner_model = GLiNER.from_pretrained("urchade/gliner_multi_pii-v1")
                log.info("gliner_model_loaded")
            except Exception as e:
                log.error("gliner_load_failed", error=str(e))
                raise
        return self._gliner_model

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

    def get_gliner_labels(self) -> List[str]:
        if self.ontology:
            return self.ontology.get_active_labels(limit=50)
        return ["Person", "Organization", "Location", "Event", "Product", "Technology"]

    def extract_batch(self, articles: List[ArticleModel]) -> List[List[EntityMention]]:
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

    def _extract_gliner_batch(self, articles: List[ArticleModel]) -> List[List[EntityMention]]:
        model = self._load_gliner()
        labels = self.get_gliner_labels()
        results = []

        for article in articles:
            text = article.content
            try:
                entities = model.predict_entities(text, labels, threshold=CONFIDENCE_THRESHOLD)
                mentions = []
                for ent in entities:
                    mentions.append(EntityMention(
                        text=ent["text"],
                        entity_type=ent["label"],
                        confidence=ent.get("score", 0.5),
                        article_id=article.id,
                        span_start=ent["start"],
                        span_end=ent["end"]
                    ))
                results.append(mentions)
            except Exception as e:
                log.warning("gliner_article_failed", article_id=article.id, error=str(e))
                results.append([])

        return results

    def _extract_spacy(self, article: ArticleModel) -> List[EntityMention]:
        nlp = self._load_spacy()
        doc = nlp(article.content)
        mentions = []

        type_map = {
            "PERSON": "Person",
            "ORG": "Organization",
            "GPE": "Location",
            "LOC": "Location",
            "EVENT": "Event",
            "PRODUCT": "Product",
            "WORK_OF_ART": "Product",
            "LAW": "Policy",
            "NORP": "Organization"
        }

        for ent in doc.ents:
            mapped_type = type_map.get(ent.label_, "Unknown")
            mentions.append(EntityMention(
                text=ent.text,
                entity_type=mapped_type,
                confidence=0.7,
                article_id=article.id,
                span_start=ent.start_char,
                span_end=ent.end_char
            ))

        return mentions
