"""
Compatibility module for relation extraction.

The pipeline now uses GLiREL from ``src.entities.entity_extractor`` directly:

    EntityExtractor.extract_with_glirel(articles)

This file intentionally contains no LLM client code and never makes network API
calls. It remains only so older imports of ``RelationExtractor`` fail softly
instead of reintroducing an LLM relation path.
"""

from typing import List

from src.models.event import EventModel
from src.models.relation import LLMRelationResponse
from src.utils.logger import get_logger

log = get_logger(__name__)


class RelationExtractor:
    """Deprecated no-LLM adapter.

    Relations are extracted in Stage 2b by GLiREL over articles, before graph
    construction. Event-level LLM extraction has been removed.
    """

    def __init__(self, *_, **__):
        log.info(
            "legacy_relation_extractor_disabled",
            replacement="EntityExtractor.extract_with_glirel",
        )

    def extract_relations(self, event: EventModel) -> LLMRelationResponse:
        log.warning(
            "legacy_relation_extraction_skipped",
            event_id=getattr(event, "event_id", ""),
            reason="LLM relation extraction has been removed; use GLiREL triples.",
        )
        return LLMRelationResponse(
            event_label="",
            triples=[],
            discovered_entity_types=[],
        )

    def extract_batch(self, events: List[EventModel]) -> List[LLMRelationResponse]:
        return [self.extract_relations(event) for event in events]
