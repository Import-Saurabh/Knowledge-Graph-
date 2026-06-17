"""
src/relations/relation_extractor.py
LLM-based relation + event-label extractor.

Rate-limit fix (root cause & solution):
  The previous _RateLimiter had a race condition: it updated _last_call
  *inside* the lock but *before* the HTTP call.  A second thread would
  acquire the lock, see the freshly-written timestamp, wait the full
  min_interval, then fire — making both calls land nearly simultaneously.
  The fix:
    1.  _last_call is updated AFTER the HTTP call returns (or raises), so
        the measured gap is the true wall-clock distance between call completions.
    2.  min_interval raised from 2.1 s → 3.5 s (~17 RPM sustained), giving
        comfortable headroom under Groq free-tier burst limits.
    3.  The lock is held for the *entire* acquire-→-call-→-release cycle via
        a reentrant design: threads queue up on the lock and the one that holds
        it owns the call slot, preventing any two calls from overlapping.
    4.  extract_batch falls back to pure sequential execution for Groq
        (max_workers forced to 1 at the provider level) so the rate limiter
        is never stressed by concurrent threads.
  All other behaviour (provider routing, JSON parsing, GLiREL triples,
  fallback LLMRelationResponse) is unchanged.
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
        self._token = object()  # sentinel — updated each cycle

    def acquire(self) -> object:
        """Block until a call slot is available.  Returns an opaque token."""
        self._lock.acquire()
        now  = time.monotonic()
        wait = self.min_interval - (now - self._last_call)
        if wait > 0:
            time.sleep(wait)
        # Return the sentinel; caller passes it back to release() as a
        # lightweight guard against mismatched acquire/release calls.
        return self._token

    def release(self, token: object) -> None:
        """Record call completion time and free the lock."""
        if token is not self._token:
            # Mismatched token — still release the lock to avoid deadlock.
            self._lock.release()
            raise RuntimeError("_RateLimiter.release() called with wrong token")
        self._last_call = time.monotonic()
        self._lock.release()


_RATE_LIMITER = _RateLimiter(min_interval=3.5)   # ~17 RPM — safe for Groq 30 RPM free tier

# ---------------------------------------------------------------------------

LLM_SYSTEM_PROMPT = """
You are a knowledge graph extractor.
Given an event context, perform two tasks:

TASK 1 - Event & Relations:
Extract a short event label and entity relation triples.
Relations should be described as natural language verbs/phrases
(e.g., "launched airstrike against", "signed trade deal with", "imposed sanctions on").
DO NOT use a fixed relation list. Describe the action in 2-5 words.

TASK 2 - Entity Type Induction:
For any entities whose type seems unclear or too generic, suggest a specific
fine-grained type.
Examples: "Biotechnology Company", "Non-State Armed Group",
          "Regional Trade Bloc", "Space Agency".

Output ONLY valid JSON:
{
  "event_label": "string",
  "triples": [
    {
      "source": "canonical entity name",
      "relation": "natural language relation phrase",
      "target": "canonical entity name",
      "confidence": 0.0-1.0
    }
  ],
  "discovered_types": [
    {
      "entity_name": "string",
      "suggested_type": "string",
      "reasoning": "string"
    }
  ]
}
"""

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

# Max workers per provider.
# Groq: forced to 1. The rate limiter now serialises calls (one in-flight at
# a time), so extra threads just queue up and add overhead without benefit.
# Other providers can run concurrent calls because _RATE_LIMITER only
# enforces a post-call gap; overlap is safe when RPM headroom exists.
_PROVIDER_MAX_WORKERS = {
    "groq":       1,   # serialised by _RATE_LIMITER — no benefit from >1
    "openrouter": 2,
    "openai":     5,
    "anthropic":  5,
    "moonshot":   3,
}


class RelationExtractor:
    def __init__(self, provider: str = None, max_workers: int = 2):
        self.provider     = (provider or settings.LLM_PROVIDER).lower().strip()
        self._client      = None
        self._model: str | None = None
        # Honour the caller's preference but never exceed the per-provider cap
        provider_cap      = _PROVIDER_MAX_WORKERS.get(self.provider, max_workers)
        self.max_workers  = min(max_workers, provider_cap)

    # ------------------------------------------------------------------
    # Config & client
    # ------------------------------------------------------------------

    def _get_config(self):
        defaults   = PROVIDER_DEFAULTS.get(self.provider, PROVIDER_DEFAULTS["groq"])
        configured = settings.LLM_MODEL
        if configured:
            is_namespaced       = "/" in configured
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
        if self._client is not None:
            return self._client

        defaults, model = self._get_config()
        self._model = model

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
        """Parse Groq 'Please try again in 24.68s' style messages."""
        m = re.search(r"try again in ([\d.]+)s", error_text, re.IGNORECASE)
        return float(m.group(1)) if m else None

    @staticmethod
    def _parse_json(content: str) -> dict:
        """Strip markdown fences then parse JSON."""
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
            "Extract the event label, relations, and any discovered entity types."
        )

        max_attempts = 5   # raised from 3

        for attempt in range(max_attempts):
            try:
                # ── Serialise via rate limiter; hold slot for entire call ─
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
                            temperature=0.3,
                        )
                        content = response.choices[0].message.content
                finally:
                    # Always release — even if the HTTP call raises — so the
                    # lock is never left held on a network error / timeout.
                    _RATE_LIMITER.release(_token)

                data = self._parse_json(content)

                triples = [
                    RelationTriple(
                        source=t.get("source", ""),
                        relation=t.get("relation", ""),
                        target=t.get("target", ""),
                        confidence=t.get("confidence", 0.5),
                        event_id=event.event_id,
                        source_article_ids=event.article_ids,
                    )
                    for t in data.get("triples", [])
                ]

                return LLMRelationResponse(
                    event_label=data.get("event_label", "Unknown Event"),
                    triples=triples,
                    discovered_entity_types=data.get("discovered_types", []),
                )

            except Exception as e:
                error_str    = str(e)
                is_rate_limit = "429" in error_str or "rate_limit" in error_str.lower()

                if is_rate_limit:
                    retry_after = self._extract_retry_after(error_str)
                    # Add 1 s buffer on top of whatever Groq says
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

        # For tiny batches just go sequential — no thread-pool overhead
        if len(events) <= 2 or self.max_workers == 1:
            return [self.extract_relations(e) for e in events]

        results: List[LLMRelationResponse | None] = [None] * len(events)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            for i, event in enumerate(events):
                future = executor.submit(self.extract_relations, event)
                futures[future] = i
                # Secondary stagger: gives the rate-limiter a head-start
                # before the next thread tries to acquire it
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