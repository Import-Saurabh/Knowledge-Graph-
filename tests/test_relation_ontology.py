import pytest
import os
from src.relations.relation_ontology import RelationOntologyManager

@pytest.fixture
def relation_ontology(tmp_path):
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path}/test.db"
    os.environ["CHROMA_PERSIST_DIR"] = str(tmp_path / "chroma")
    from src.utils.db import init_db
    init_db()
    from src.vectorstore.chroma_manager import ChromaManager
    from src.embeddings.embedding_generator import EmbeddingGenerator
    chroma = ChromaManager(str(tmp_path / "chroma"))
    embedder = EmbeddingGenerator()
    return RelationOntologyManager(chroma, embedder)

def test_free_form_relation_accepted(relation_ontology):
    rom = relation_ontology
    canonical = rom.normalize_relation("levied export controls on")
    assert canonical is not None
    assert len(canonical) > 0

def test_relation_clustering(relation_ontology):
    rom = relation_ontology
    r1 = rom.normalize_relation("launched airstrike against")
    r2 = rom.normalize_relation("conducted bombing raid on")
    r3 = rom.normalize_relation("signed trade deal with")

    clusters = rom.cluster_relations()
    assert isinstance(clusters, dict)
