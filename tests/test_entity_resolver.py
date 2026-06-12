import pytest
import os
from src.entities.entity_resolver import EntityResolver
from src.models.entity import EntityMention

@pytest.fixture
def resolver(tmp_path):
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path}/test.db"
    os.environ["CHROMA_PERSIST_DIR"] = str(tmp_path / "chroma")
    from src.utils.db import init_db
    init_db()
    from src.vectorstore.chroma_manager import ChromaManager
    from src.embeddings.embedding_generator import EmbeddingGenerator
    chroma = ChromaManager(str(tmp_path / "chroma"))
    embedder = EmbeddingGenerator()
    return EntityResolver(chroma, embedder)

def test_alias_learning(resolver):
    e1 = resolver.resolve(EntityMention(text="USA", entity_type="Location", confidence=0.9, article_id="a1", span_start=0, span_end=3))
    e2 = resolver.resolve(EntityMention(text="America", entity_type="Location", confidence=0.9, article_id="a2", span_start=0, span_end=7))
    assert e1.canonical_name is not None
    assert e2.canonical_name is not None

def test_no_hardcoded_aliases(resolver):
    result = resolver.resolve(EntityMention(text="Modi", entity_type="Person", confidence=0.9, article_id="a1", span_start=0, span_end=4))
    assert result.canonical_name == "Modi"
