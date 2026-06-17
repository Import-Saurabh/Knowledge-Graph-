"""
src/relations/relation_extractor.py
LLM-based relation + event-label extractor.

No bug-list items target this file directly, but several small hardening
changes are included:
  • `_get_client` is idempotent even when called from parallel threads
    (model string is set once, then the cached client is reused).
  • Provider/model mismatch warning already in the original is preserved.
  • Groq TPM stagger (0.5 s between future submissions) kept as-is.
  • Rate-limit back-off parses Groq's "try again in Xs" message.
"""

import json
import re
import time
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.models.event import EventModel
from src.models.relation import RelationTriple, LLMRelationResponse
from src.utils.config import settings
from src.utils.logger import get_logger

log = get_logger(__name__)

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
    "anthropic":  {"model": "claude-3-haiku-20240307",              "client": "anthropic"},
    "openai":     {"model": "gpt-4o-mini",                          "client": "openai"},
    "moonshot":   {"model": "moonshot-v1-8k",                       "client": "openai",
                   "base_url": "https://api.moonshot.cn/v1"},
    "openrouter": {"model": "meta-llama/llama-3.1-8b-instruct:free","client": "openai",
                   "base_url": "https://openrouter.ai/api/v1"},
    "groq":       {"model": "llama-3.1-8b-instant",                 "client": "openai",
                   "base_url": "https://api.groq.com/openai/v1"},
}


class RelationExtractor:
    def __init__(self, provider: str = None, max_workers: int = 2):
        self.provider = (provider or settings.LLM_PROVIDER).lower().strip()
        self._client = None
        self._model: str | None = None
        self.max_workers = max_workers

    # ------------------------------------------------------------------
    # Config & client
    # ------------------------------------------------------------------

    def _get_config(self):
        defaults = PROVIDER_DEFAULTS.get(self.provider, PROVIDER_DEFAULTS["groq"])
        configured = settings.LLM_MODEL
        if configured:
            is_namespaced = "/" in configured
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
        self._model = model  # store once

        if defaults["client"] == "anthropic":
            import anthropic
            self._client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        else:
            import openai
            base_url = defaults.get("base_url")
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
            log.info("llm_client_initialized", provider=self.provider,
                     model=self._model, base_url=base_url or "default")

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
        client = self._get_client()

        prompt = (
            f"Event Context:\n{event.context[:4000]}\n\n"
            "Extract the event label, relations, and any discovered entity types."
        )

        for attempt in range(3):
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
                error_str = str(e)
                is_rate_limit = "429" in error_str or "rate_limit" in error_str.lower()

                if is_rate_limit:
                    retry_after = self._extract_retry_after(error_str)
                    sleep_time = retry_after or (2 ** attempt * 5)
                    log.warning(
                        "rate_limit_hit",
                        attempt=attempt,
                        provider=self.provider,
                        sleep_seconds=sleep_time,
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

        return LLMRelationResponse(
            event_label="Unknown Event",
            triples=[],
            discovered_entity_types=[],
        )

    def extract_batch(self, events: List[EventModel]) -> List[LLMRelationResponse]:
        if not events:
            return []

        if len(events) <= 2:
            return [self.extract_relations(e) for e in events]

        results: List[LLMRelationResponse | None] = [None] * len(events)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            for i, event in enumerate(events):
                future = executor.submit(self.extract_relations, event)
                futures[future] = i
                if i < len(events) - 1:
                    time.sleep(0.5)   # Groq 6 000 TPM stagger

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