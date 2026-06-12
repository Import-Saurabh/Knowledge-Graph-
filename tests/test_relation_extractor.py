import pytest
from src.relations.relation_extractor import RelationExtractor

def test_relation_extractor_init():
    extractor = RelationExtractor()
    assert extractor.provider is not None
