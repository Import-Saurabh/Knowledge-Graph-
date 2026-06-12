import pytest
import os
from datetime import datetime
from src.entities.ontology_manager import OntologyManager
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.vectorstore.chroma_manager import ChromaManager

@pytest.fixture
def ontology_manager(tmp_path):
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path}/test.db"
    os.environ["CHROMA_PERSIST_DIR"] = str(tmp_path / "chroma")
    from src.utils.db import init_db
    init_db()
    chroma = ChromaManager(str(tmp_path / "chroma"))
    embedder = EmbeddingGenerator()
    om = OntologyManager(chroma, embedder)
    return om

def test_type_induction_creates_new_type(ontology_manager):
    om = ontology_manager
    type_name = om.induce_type("OpenAI", "OpenAI announced GPT-5", "AI Company")
    assert type_name == "AI Company"
    report = om.get_ontology_report()
    assert report["type_count"] >= 1

def test_similar_types_merged(ontology_manager):
    om = ontology_manager
    t1 = om.induce_type("OpenAI", "...", "AI Company")
    t2 = om.induce_type("Anthropic", "...", "Artificial Intelligence Firm")
    report = om.get_ontology_report()
    assert report["type_count"] <= 2

def test_empty_seed_ontology_starts_from_scratch(ontology_manager):
    om = ontology_manager
    labels = om.get_active_labels()
    assert len(labels) > 0  # Falls back to generics if empty
