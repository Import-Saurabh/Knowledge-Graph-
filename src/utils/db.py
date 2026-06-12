from sqlalchemy import create_engine, Column, String, DateTime, Integer, Float, Boolean, Text, ForeignKey, JSON, BLOB, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import enum
import json

Base = declarative_base()

class ArticleStatus(str, enum.Enum):
    ingested = "ingested"
    embedded = "embedded"
    deduplicated = "deduplicated"
    clustered = "clustered"
    llm_done = "llm_done"
    graph_built = "graph_built"

class ArticleDB(Base):
    __tablename__ = "articles"
    id = Column(String, primary_key=True)
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    source = Column(String, nullable=False)
    published_at = Column(DateTime, nullable=False)
    url = Column(String, nullable=False)
    status = Column(String, default="ingested")
    duplicate_group_id = Column(String, nullable=True)
    cluster_id = Column(Integer, nullable=True)
    temporal_window = Column(String, nullable=True)
    embedding_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class EventDB(Base):
    __tablename__ = "events"
    event_id = Column(String, primary_key=True)
    cluster_id = Column(Integer, nullable=False)
    temporal_window = Column(String, nullable=False)
    article_ids = Column(JSON, default=list)
    representative_article_ids = Column(JSON, default=list)
    context = Column(Text, default="")
    entities = Column(JSON, default=list)
    llm_processed = Column(Boolean, default=False)
    graph_inserted = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class EntityOntologyDB(Base):
    __tablename__ = "entity_ontology"
    type_id = Column(String, primary_key=True)
    type_name = Column(String, unique=True, nullable=False)
    type_embedding = Column(BLOB, nullable=True)
    parent_type_id = Column(String, ForeignKey("entity_ontology.type_id"), nullable=True)
    mention_count = Column(Integer, default=0)
    first_seen = Column(DateTime, default=datetime.utcnow)
    last_seen = Column(DateTime, default=datetime.utcnow)
    is_auto_discovered = Column(Boolean, default=True)
    is_confirmed = Column(Boolean, default=False)

class CanonicalEntityDB(Base):
    __tablename__ = "canonical_entities"
    canonical_id = Column(String, primary_key=True)
    canonical_name = Column(String, unique=True, nullable=False)
    type_id = Column(String, ForeignKey("entity_ontology.type_id"), nullable=True)
    aliases = Column(Text, default="[]")
    embedding_vector = Column(BLOB, nullable=True)
    mention_count = Column(Integer, default=0)
    first_seen = Column(DateTime, default=datetime.utcnow)
    last_seen = Column(DateTime, default=datetime.utcnow)

class RelationOntologyDB(Base):
    __tablename__ = "relation_ontology"
    relation_id = Column(String, primary_key=True)
    relation_text = Column(String, unique=True, nullable=False)
    relation_canonical = Column(String, nullable=True)
    relation_embedding = Column(BLOB, nullable=True)
    cluster_id = Column(Integer, nullable=True)
    usage_count = Column(Integer, default=0)
    first_seen = Column(DateTime, default=datetime.utcnow)
    last_seen = Column(DateTime, default=datetime.utcnow)
    is_auto_discovered = Column(Boolean, default=True)

class EntityAliasDB(Base):
    __tablename__ = "entity_aliases"
    alias_id = Column(String, primary_key=True)
    alias_text = Column(String, nullable=False)
    canonical_id = Column(String, ForeignKey("canonical_entities.canonical_id"), nullable=False)
    match_score = Column(Float, nullable=True)
    source_article_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class MentionDB(Base):
    __tablename__ = "mentions"
    mention_id = Column(String, primary_key=True)
    text = Column(String, nullable=False)
    entity_type = Column(String, nullable=False)
    confidence = Column(Float, default=0.0)
    article_id = Column(String, ForeignKey("articles.id"), nullable=False)
    canonical_id = Column(String, ForeignKey("canonical_entities.canonical_id"), nullable=True)
    span_start = Column(Integer, nullable=False)
    span_end = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

_engine = None
_session_factory = None

def get_engine():
    global _engine
    if _engine is None:
        from src.utils.config import settings
        _engine = create_engine(settings.DATABASE_URL, echo=False)
    return _engine

def get_session():
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(bind=get_engine())
    return _session_factory()

def init_db():
    Base.metadata.create_all(get_engine())
