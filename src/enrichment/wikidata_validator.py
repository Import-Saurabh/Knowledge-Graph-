import logging
import urllib.error
from typing import Optional, Dict, Any

from SPARQLWrapper import SPARQLWrapper, JSON

log = logging.getLogger(__name__)

_EMPTY_VALIDATION = {
    "exists": False,
    "wikidata_property": None,
    "confidence_boost": 0.0,
    "description": None,
    "date": None,
}


class WikidataValidator:
    def __init__(self, cache_size: int = 10000, timeout: int = 10):
        self.sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
        self.sparql.setReturnFormat(JSON)
        self.sparql.setTimeout(timeout)
        self._entity_cache: Dict[str, Dict] = {}
        self._relation_cache: Dict[tuple, Dict] = {}
        self._qid_cache: Dict[str, Optional[str]] = {}
        # Circuit breaker: tripped on first 429; prevents all further SPARQL calls.
        self._rate_limited: bool = False

    def _query(self, query: str) -> list:
        """
        Execute a SPARQL query and return result bindings.

        Returns [] on any error. On HTTP 429, trips the circuit breaker so
        all subsequent calls return [] immediately without hitting the endpoint.
        """
        if self._rate_limited:
            return []
        try:
            self.sparql.setQuery(query)
            results = self.sparql.query().convert()
            return results["results"]["bindings"]
        except urllib.error.HTTPError as e:
            if e.code == 429:
                self._rate_limited = True
                log.warning(
                    "Wikidata SPARQL returned 429 (rate-limited). "
                    "All further Wikidata queries are disabled for this run. "
                    "Re-run with --skip-wikidata to bypass Stage 9b entirely."
                )
            else:
                log.warning("SPARQL HTTP error %s: %s", e.code, e)
            return []
        except Exception as e:
            log.warning("SPARQL query failed: %s", e)
            return []

    @property
    def is_available(self) -> bool:
        """False once a 429 has been received; callers can check before looping."""
        return not self._rate_limited

    def get_wikidata_id(self, entity_name: str) -> Optional[str]:
        """Resolve entity name to Wikidata Q-ID (exact match)."""
        if entity_name in self._qid_cache:
            return self._qid_cache[entity_name]

        query = f"""
        SELECT ?item WHERE {{
          ?item rdfs:label "{entity_name}"@en.
        }}
        LIMIT 1
        """
        results = self._query(query)
        qid = results[0]["item"]["value"].split("/")[-1] if results else None
        self._qid_cache[entity_name] = qid
        return qid

    def validate_relation(self, subject: str, relation: str, obj: str) -> Dict[str, Any]:
        """
        Check if a relation exists in Wikidata between subject and object.

        Returns dict with keys:
          exists: bool
          wikidata_property: str (P-ID) or None
          confidence_boost: float (0.0 to 0.3)
          description: str or None
          date: str (ISO date) or None
        """
        if self._rate_limited:
            return dict(_EMPTY_VALIDATION)

        cache_key = (subject, relation, obj)
        if cache_key in self._relation_cache:
            return self._relation_cache[cache_key]

        subj_id = self.get_wikidata_id(subject)
        obj_id  = self.get_wikidata_id(obj)

        if not subj_id or not obj_id:
            result = dict(_EMPTY_VALIDATION)
            self._relation_cache[cache_key] = result
            return result

        # Forward direction
        query = f"""
        SELECT ?property ?propertyLabel ?date WHERE {{
          wd:{subj_id} ?property wd:{obj_id}.
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
          OPTIONAL {{ wd:{subj_id} ?property ?date. FILTER(DATATYPE(?date) = xsd:dateTime) }}
        }}
        LIMIT 10
        """
        results = self._query(query)
        if results:
            prop       = results[0]["property"]["value"].split("/")[-1]
            prop_label = results[0].get("propertyLabel", {}).get("value", "")
            date       = results[0].get("date", {}).get("value")
            result = {
                "exists": True,
                "wikidata_property": prop,
                "confidence_boost": 0.3,
                "description": f"Wikidata property: {prop_label}",
                "date": date,
            }
            self._relation_cache[cache_key] = result
            return result

        # Reverse direction
        query_rev = f"""
        SELECT ?property ?propertyLabel ?date WHERE {{
          wd:{obj_id} ?property wd:{subj_id}.
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
          OPTIONAL {{ wd:{obj_id} ?property ?date. FILTER(DATATYPE(?date) = xsd:dateTime) }}
        }}
        LIMIT 10
        """
        results = self._query(query_rev)
        if results:
            prop       = results[0]["property"]["value"].split("/")[-1]
            prop_label = results[0].get("propertyLabel", {}).get("value", "")
            date       = results[0].get("date", {}).get("value")
            result = {
                "exists": True,
                "wikidata_property": prop,
                "confidence_boost": 0.2,
                "description": f"Wikidata property (reverse): {prop_label}",
                "date": date,
            }
            self._relation_cache[cache_key] = result
            return result

        result = dict(_EMPTY_VALIDATION)
        self._relation_cache[cache_key] = result
        return result

    def enrich_entity(self, entity_name: str) -> Dict[str, Any]:
        """
        Fetch description, types, and Wikidata ID for an entity.

        Returns dict with keys:
          wikidata_id: str (Q-ID)
          description: str
          types: list of type labels
          url: str (URL to Wikidata page)
        """
        if self._rate_limited:
            return {}

        if entity_name in self._entity_cache:
            return self._entity_cache[entity_name]

        qid = self.get_wikidata_id(entity_name)
        if not qid:
            self._entity_cache[entity_name] = {}
            return {}

        query = f"""
        SELECT ?description ?type ?typeLabel WHERE {{
          wd:{qid} schema:description ?description.
          FILTER(LANG(?description) = "en")
          OPTIONAL {{ wd:{qid} wdt:P31 ?type. }}
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        LIMIT 10
        """
        results = self._query(query)
        if not results:
            self._entity_cache[entity_name] = {}
            return {}

        desc  = results[0].get("description", {}).get("value", "")
        types = list({
            row["typeLabel"]["value"]
            for row in results
            if "typeLabel" in row
        })
        info = {
            "wikidata_id": qid,
            "description": desc,
            "types": types,
            "url": f"https://www.wikidata.org/wiki/{qid}",
        }
        self._entity_cache[entity_name] = info
        return info