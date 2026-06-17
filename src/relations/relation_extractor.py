"""
src/relations/relation_extractor.py

LLM-based relation + event-label extractor.

Changes vs previous version:
  - LLM_SYSTEM_PROMPT now instructs the model to pick from a predefined
    controlled vocabulary (CANONICAL_RELATION_LIST).  Both the canonical
    label AND the original raw phrase are returned in every triple so the
    graph stores both: raw for human readability, canonical for querying.
  - Triple parsing handles the new `relation_raw` field and populates
    RelationTriple.relation (raw) + RelationTriple.relation_canonical.
  - If the model returns a label that is NOT in the vocabulary the extractor
    validates it in _validate_canonical(); unknown labels are remapped via
    the relation_ontology at call-site rather than silently stored as-is.
  - All rate-limit / retry / provider-routing logic is unchanged.
"""

import json
import re
import threading
import time
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.models.event import EventModel
from src.models.relation import RelationTriple, LLMRelationResponse
from src.utils.config import settings
from src.utils.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Module-level rate limiter — shared across ALL threads and ALL instances
# ---------------------------------------------------------------------------

class _RateLimiter:
    """
    Serialises LLM calls so that the wall-clock gap between the *end* of one
    call and the *start* of the next is at least `min_interval` seconds.

    Key design decisions
    --------------------
    * The lock is acquired BEFORE the sleep and held until AFTER the caller's
      HTTP round-trip is complete (caller must call release()).  This means
      threads queue up strictly; there is never more than one in-flight call.
    * _last_call is written AFTER the call, not before, so the measured gap
      reflects actual network time, not an optimistic pre-timestamp.
    * min_interval=3.5 s → ~17 RPM sustained — well within Groq free-tier
      30 RPM but safe against burst-window enforcement.

    Usage
    -----
        token = _RATE_LIMITER.acquire()   # blocks until slot is free
        try:
            result = make_http_call()
        finally:
            _RATE_LIMITER.release(token)  # records completion time & frees lock
    """

    def __init__(self, min_interval: float = 3.5):
        self._lock = threading.Lock()
        self._last_call: float = 0.0
        self.min_interval = min_interval
        self._token = object()

    def acquire(self) -> object:
        self._lock.acquire()
        now  = time.monotonic()
        wait = self.min_interval - (now - self._last_call)
        if wait > 0:
            time.sleep(wait)
        return self._token

    def release(self, token: object) -> None:
        if token is not self._token:
            self._lock.release()
            raise RuntimeError("_RateLimiter.release() called with wrong token")
        self._last_call = time.monotonic()
        self._lock.release()


_RATE_LIMITER = _RateLimiter(min_interval=3.5)


# ---------------------------------------------------------------------------
# Controlled vocabulary — MUST stay in sync with relation_ontology.py
# The LLM is instructed to pick exactly one label from this list.
# ---------------------------------------------------------------------------

CANONICAL_RELATION_LIST: List[str] = [
    # Military / Conflict
    "MILITARY_ATTACK",
    "MILITARY_OCCUPATION",
    "MILITARY_WITHDRAWAL",
    "MILITARY_SUPPORT",
    "CEASEFIRE",
    # Diplomatic
    "DIPLOMATIC_MEETING",
    "DIPLOMATIC_AGREEMENT",
    "DIPLOMATIC_RECOGNITION",
    "DIPLOMATIC_EXPULSION",
    "DIPLOMATIC_STATEMENT",
    "PEACE_NEGOTIATION",
    # Sanctions / Economic
    "SANCTIONS_IMPOSED",
    "SANCTIONS_LIFTED",
    "TRADE_AGREEMENT",
    "ECONOMIC_AID",
    "INVESTMENT",
    # Leadership / Political
    "LEADER_OF",
    "APPOINTED",
    "RESIGNED",
    "ALLY_OF",
    "OPPOSES",
    # Legal / Criminal
    "ACCUSED_OF",
    "CONVICTED_OF",
    "ARRESTED",
    # Organisational
    "MEMBER_OF",
    "FOUNDED",
    "HEADQUARTERED_IN",
    # Humanitarian / Population
    "HUMANITARIAN_AID",
    "REFUGEE_MOVEMENT",
    # Nuclear / Security
    "NUCLEAR_ACTIVITY",
    # Generic fallback — use sparingly
    "RELATED_TO",
]

_CANONICAL_SET = set(CANONICAL_RELATION_LIST)

# ---------------------------------------------------------------------------
# System prompt — enforces controlled vocabulary at the LLM level
# ---------------------------------------------------------------------------

_VOCAB_BLOCK = "\n".join(f"  - {r}" for r in CANONICAL_RELATION_LIST)

LLM_SYSTEM_PROMPT = f"""
You are a knowledge graph extractor for news and geopolitical events.
Given an event context, perform two tasks.

══════════════════════════════════════════════════════════════
TASK 1 — Event label and relation triples
══════════════════════════════════════════════════════════════
Extract a short event label (≤ 8 words) and entity relation triples.

For each triple you MUST:
  1. Write the raw phrase that best describes the relation in the text
     (2–5 natural-language words, e.g. "launched airstrikes against").
     Store this in the `relation` field — it is kept for human readability.
  2. Pick the single BEST-MATCHING canonical label from the list below
     and store it in `relation_canonical`.
     Use "RELATED_TO" only if no other label clearly fits.

CANONICAL RELATION VOCABULARY (pick exactly one per triple):
{_VOCAB_BLOCK}

══════════════════════════════════════════════════════════════
TASK 2 — Entity type induction
══════════════════════════════════════════════════════════════
For any entity whose type seems too generic, suggest a fine-grained type.
Examples: "Biotechnology Company", "Non-State Armed Group",
          "Regional Trade Bloc", "Space Agency".

══════════════════════════════════════════════════════════════
OUTPUT — valid JSON only, no markdown fences
══════════════════════════════════════════════════════════════
{{
  "event_label": "string (≤ 8 words)",
  "triples": [
    {{
      "source":             "canonical entity name",
      "relation":           "raw natural-language phrase (2–5 words)",
      "relation_canonical": "ONE label from the vocabulary above",
      "target":             "canonical entity name",
      "confidence":         0.0–1.0
    }}
  ],
  "discovered_types": [
    {{
      "entity_name":    "string",
      "suggested_type": "string",
      "reasoning":      "string"
    }}
  ]
}}
"""

# ---------------------------------------------------------------------------
# Provider defaults
# ---------------------------------------------------------------------------

PROVIDER_DEFAULTS = {
    "anthropic":  {"model": "claude-3-haiku-20240307",               "client": "anthropic"},
    "openai":     {"model": "gpt-4o-mini",                           "client": "openai"},
    "moonshot":   {"model": "moonshot-v1-8k",                        "client": "openai",
                   "base_url": "https://api.moonshot.cn/v1"},
    "openrouter": {"model": "meta-llama/llama-3.1-8b-instruct:free", "client": "openai",
                   "base_url": "https://openrouter.ai/api/v1"},
    "groq":       {"model": "llama-3.1-8b-instant",                  "client": "openai",
                   "base_url": "https://api.groq.com/openai/v1"},
}

_PROVIDER_MAX_WORKERS = {
    "groq":       1,
    "openrouter": 2,
    "openai":     5,
    "anthropic":  5,
    "moonshot":   3,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_canonical(label: str) -> str:
    """
    Return label if it is in the vocabulary.
    Otherwise return "RELATED_TO" so the graph never stores raw LLM noise
    as a canonical.  The ontology manager will remap it properly later.
    """
    if not label:
        return "RELATED_TO"
    normalized = label.strip().upper().replace(" ", "_")
    return normalized if normalized in _CANONICAL_SET else "RELATED_TO"


# ---------------------------------------------------------------------------

class RelationExtractor:
    def __init__(self, provider: str = None, max_workers: int = 2):
        self.provider    = (provider or settings.LLM_PROVIDER).lower().strip()
        self._client     = None
        self._model: str | None = None
        provider_cap     = _PROVIDER_MAX_WORKERS.get(self.provider, max_workers)
        self.max_workers = min(max_workers, provider_cap)

    # ------------------------------------------------------------------
    # Config & client
    # ------------------------------------------------------------------

    def _get_config(self):
        defaults   = PROVIDER_DEFAULTS.get(self.provider, PROVIDER_DEFAULTS["groq"])
        configured = settings.LLM_MODEL
        if configured:
            is_namespaced        = "/" in configured
            provider_needs_plain = self.provider in ("groq", "openai", "anthropic", "moonshot")
            if provider_needs_plain and is_namespaced:
                log.warning(
                    "model_provider_mismatch",
                    configured_model=configured,
                    provider=self.provider,
                    fallback_model=defaults["model"],
                    hint="Set LLM_MODEL to a plain slug for this provider, or leave it blank.",
                )
                model = defaults["model"]
            else:
                model = configured
        else:
            model = defaults["model"]
        return defaults, model

    def _get_client(self):
        if self._client:
            return self._client

        defaults, model = self._get_config()
        self._model     = model

        if defaults["client"] == "anthropic":
            import anthropic
            self._client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        else:
            import openai
            base_url    = defaults.get("base_url")
            api_key_map = {
                "openai":     settings.OPENAI_API_KEY,
                "moonshot":   settings.MOONSHOT_API_KEY,
                "openrouter": settings.OPENROUTER_API_KEY,
                "groq":       settings.GROQ_API_KEY,
            }
            kwargs = {"api_key": api_key_map.get(self.provider, settings.OPENAI_API_KEY)}
            if base_url:
                kwargs["base_url"] = base_url
            self._client = openai.OpenAI(**kwargs)
            log.info("llm_client_initialized",
                     provider=self.provider, model=self._model,
                     base_url=base_url or "default",
                     effective_workers=self.max_workers)

        return self._client

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_retry_after(error_text: str) -> float | None:
        m = re.search(r"try again in ([\d.]+)s", error_text, re.IGNORECASE)
        return float(m.group(1)) if m else None

    @staticmethod
    def _parse_json(content: str) -> dict:
        text = content
        if "```json" in text:
            text = text.split("```json", 1)[1].split("```", 1)[0]
        elif "```" in text:
            text = text.split("```", 1)[1].split("```", 1)[0]
        return json.loads(text.strip())

    # ------------------------------------------------------------------
    # Core extraction
    # ------------------------------------------------------------------

    def extract_relations(self, event: EventModel) -> LLMRelationResponse:
        defaults, model = self._get_config()
        client          = self._get_client()

        prompt = (
            f"Event Context:\n{event.context[:4000]}\n\n"
            "Extract the event label, relations (with canonical labels), "
            "and any discovered entity types."
        )

        max_attempts = 5

        for attempt in range(max_attempts):
            try:
                _token = _RATE_LIMITER.acquire()
                try:
                    if defaults["client"] == "anthropic":
                        response = client.messages.create(
                            model=model,
                            max_tokens=2000,
                            system=LLM_SYSTEM_PROMPT,
                            messages=[{"role": "user", "content": prompt}],
                        )
                        content = response.content[0].text
                    else:
                        response = client.chat.completions.create(
                            model=model,
                            messages=[
                                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                                {"role": "user",   "content": prompt},
                            ],
                            max_tokens=2000,
                            temperature=0.2,   # lower than before — vocabulary compliance
                        )
                        content = response.choices[0].message.content
                finally:
                    _RATE_LIMITER.release(_token)

                data = self._parse_json(content)

                # ── Parse triples ────────────────────────────────────────
                triples = []
                for t in data.get("triples", []):
                    source   = (t.get("source")   or "").strip()
                    relation  = (t.get("relation") or "").strip()
                    target   = (t.get("target")   or "").strip()

                    if not source or not relation or not target:
                        log.warning(
                            "skipping_invalid_triple",
                            event_id=event.event_id,
                            source=source,
                            relation=relation,
                            target=target,
                        )
                        continue

                    confidence = t.get("confidence", 0.5)
                    if not isinstance(confidence, (int, float)):
                        confidence = 0.5

                    # relation_canonical: validate against vocab; remap if needed
                    raw_canonical = (t.get("relation_canonical") or "").strip()
                    canonical     = _validate_canonical(raw_canonical)

                    if raw_canonical and raw_canonical.upper() not in _CANONICAL_SET:
                        log.debug(
                            "canonical_remapped",
                            event_id=event.event_id,
                            llm_returned=raw_canonical,
                            remapped_to=canonical,
                        )

                    triples.append(
                        RelationTriple(
                            source=source,
                            relation=relation,            # raw phrase — preserved
                            relation_canonical=canonical, # controlled-vocab label
                            target=target,
                            confidence=confidence,
                            event_id=event.event_id,
                            source_article_ids=event.article_ids,
                        )
                    )

                return LLMRelationResponse(
                    event_label=data.get("event_label", "Unknown Event"),
                    triples=triples,
                    discovered_entity_types=data.get("discovered_types", []),
                )

            except Exception as e:
                error_str     = str(e)
                is_rate_limit = "429" in error_str or "rate_limit" in error_str.lower()

                if is_rate_limit:
                    retry_after = self._extract_retry_after(error_str)
                    sleep_time  = (retry_after + 1.0) if retry_after else (2 ** attempt * 5)
                    log.warning(
                        "rate_limit_hit",
                        attempt=attempt,
                        provider=self.provider,
                        sleep_seconds=round(sleep_time, 2),
                        event_id=event.event_id,
                    )
                    time.sleep(sleep_time)
                else:
                    log.warning(
                        "llm_extraction_failed",
                        attempt=attempt,
                        provider=self.provider,
                        error=error_str,
                    )
                    time.sleep(1)

        log.error("extract_relations_exhausted",
                  event_id=event.event_id, attempts=max_attempts)
        return LLMRelationResponse(
            event_label="Unknown Event",
            triples=[],
            discovered_entity_types=[],
        )

    def extract_batch(self, events: List[EventModel]) -> List[LLMRelationResponse]:
        if not events:
            return []

        if len(events) <= 2 or self.max_workers == 1:
            return [self.extract_relations(e) for e in events]

        results: List[LLMRelationResponse | None] = [None] * len(events)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            for i, event in enumerate(events):
                future = executor.submit(self.extract_relations, event)
                futures[future] = i
                if i < len(events) - 1:
                    time.sleep(0.5)

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    log.warning(
                        "batch_extraction_failed",
                        event_id=events[idx].event_id,
                        idx=idx,
                        error=str(e),
                    )
                    results[idx] = LLMRelationResponse(
                        event_label="Unknown Event",
                        triples=[],
                        discovered_entity_types=[],
                    )

        return results