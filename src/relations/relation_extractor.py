import json
import time
from typing import List
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
For any entities whose type seems unclear or too generic, suggest a specific fine-grained type.
Examples: "Biotechnology Company", "Non-State Armed Group", "Regional Trade Bloc", "Space Agency".

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

# Provider defaults
PROVIDER_DEFAULTS = {
    "anthropic": {"model": "claude-3-haiku-20240307", "client": "anthropic"},
    "openai": {"model": "gpt-4o-mini", "client": "openai"},
    "moonshot": {"model": "moonshot-v1-8k", "client": "openai", "base_url": "https://api.moonshot.cn/v1"},
    "openrouter": {"model": "meta-llama/llama-3.1-8b-instruct:free", "client": "openai", "base_url": "https://openrouter.ai/api/v1"},
    "groq": {"model": "llama-3.1-8b-instant", "client": "openai", "base_url": "https://api.groq.com/openai/v1"},
}

class RelationExtractor:
    def __init__(self, provider: str = None):
        self.provider = (provider or settings.LLM_PROVIDER).lower().strip()
        self._client = None
        self._model = None

    def _get_config(self):
        defaults = PROVIDER_DEFAULTS.get(self.provider, PROVIDER_DEFAULTS["groq"])
        model = settings.LLM_MODEL or defaults["model"]
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
            # OpenAI-compatible clients: openai, moonshot, openrouter, groq
            import openai
            base_url = defaults.get("base_url")

            if self.provider == "openai":
                api_key = settings.OPENAI_API_KEY
            elif self.provider == "moonshot":
                api_key = settings.MOONSHOT_API_KEY
            elif self.provider == "openrouter":
                api_key = settings.OPENROUTER_API_KEY
            elif self.provider == "groq":
                api_key = settings.GROQ_API_KEY
            else:
                api_key = settings.OPENAI_API_KEY

            kwargs = {"api_key": api_key}
            if base_url:
                kwargs["base_url"] = base_url

            self._client = openai.OpenAI(**kwargs)
            log.info("llm_client_initialized", provider=self.provider, model=self._model, base_url=base_url or "default")

        return self._client

    def extract_relations(self, event: EventModel) -> LLMRelationResponse:
        defaults, model = self._get_config()
        client = self._get_client()

        prompt = f"Event Context:\n{event.context[:4000]}\n\nExtract the event label, relations, and any discovered entity types."

        for attempt in range(3):
            try:
                if defaults["client"] == "anthropic":
                    response = client.messages.create(
                        model=model,
                        max_tokens=2000,
                        system=LLM_SYSTEM_PROMPT,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    content = response.content[0].text
                else:
                    # OpenAI-compatible API
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": LLM_SYSTEM_PROMPT},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=2000,
                        temperature=0.3
                    )
                    content = response.choices[0].message.content

                # Extract JSON
                json_str = content
                if "```json" in content:
                    json_str = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    json_str = content.split("```")[1].split("```")[0].strip()

                data = json.loads(json_str)

                triples = []
                for t in data.get("triples", []):
                    triples.append(RelationTriple(
                        source=t.get("source", ""),
                        relation=t.get("relation", ""),
                        target=t.get("target", ""),
                        confidence=t.get("confidence", 0.5),
                        event_id=event.event_id,
                        source_article_ids=event.article_ids
                    ))

                return LLMRelationResponse(
                    event_label=data.get("event_label", "Unknown Event"),
                    triples=triples,
                    discovered_entity_types=data.get("discovered_types", [])
                )

            except Exception as e:
                log.warning("llm_extraction_failed", attempt=attempt, provider=self.provider, error=str(e))
                time.sleep(1)

        # Fallback
        return LLMRelationResponse(
            event_label="Unknown Event",
            triples=[],
            discovered_entity_types=[]
        )

    def extract_batch(self, events: List[EventModel]) -> List[LLMRelationResponse]:
        results = []
        for event in events:
            result = self.extract_relations(event)
            results.append(result)
            time.sleep(1)  # Rate limiting
        return results
