"""
src/ingestion/infobox_extractor.py

Extracts structured triples from Wikipedia infoboxes.
Now supports direct extraction from wikitext (no API calls) via `extract_from_wikitext`.
"""

import re
import time
import requests
from typing import List, Dict, Optional, Tuple
from src.utils.logger import get_logger

log = get_logger(__name__)

# Mapping from infobox field names to canonical relation labels
INFOBOX_RELATION_MAP = {
    "name": "NAME",
    "official_name": "OFFICIAL_NAME",
    "native_name": "NATIVE_NAME",
    "image": "IMAGE",
    "caption": "CAPTION",
    "image_map": "MAP",
    "map_caption": "MAP_CAPTION",
    "coordinates": "COORDINATES",
    "coords": "COORDINATES",
    "latd": "LATITUDE",
    "longd": "LONGITUDE",
    "leader_title": "LEADER_TITLE",
    "leader_name": "LEADER_NAME",
    "leader_party": "LEADER_PARTY",
    "chief_judge": "CHIEF_JUDGE",
    "area_total_km2": "AREA_TOTAL",
    "area_total": "AREA_TOTAL",
    "population_estimate": "POPULATION_ESTIMATE",
    "population_census": "POPULATION_CENSUS",
    "population_density_km2": "POPULATION_DENSITY",
    "population_density": "POPULATION_DENSITY",
    "GDP_nominal": "GDP_NOMINAL",
    "GDP_PPP": "GDP_PPP",
    "Gini": "GINI",
    "HDI": "HDI",
    "currency": "CURRENCY",
    "time_zone": "TIME_ZONE",
    "utc_offset": "UTC_OFFSET",
    "date_format": "DATE_FORMAT",
    "drives_on": "DRIVES_ON",
    "calling_code": "CALLING_CODE",
    "iso_code": "ISO_CODE",
    "internet_tld": "TLD",
    "established_date": "ESTABLISHED_DATE",
    "independence_date": "INDEPENDENCE_DATE",
    "official_language": "OFFICIAL_LANGUAGE",
    "languages": "LANGUAGES",
    "religion": "RELIGION",
    "demonym": "DEMONYM",
    "government_type": "GOVERNMENT_TYPE",
    "capital": "CAPITAL",
    "largest_city": "LARGEST_CITY",
    "admin_center": "ADMIN_CENTER",
    "subdivision_name": "SUBDIVISION_OF",
    "subdivision_type": "SUBDIVISION_TYPE",
    "founder": "FOUNDER",
    "founded_by": "FOUNDER",
    "established": "ESTABLISHED",
    "incorporation": "INCORPORATION",
    "extinction": "EXTINCTION",
    "leader": "LEADER",
    "governor": "GOVERNOR",
    "mayor": "MAYOR",
    "premier": "PREMIER",
    "prime_minister": "PRIME_MINISTER",
    "president": "PRESIDENT",
    "chancellor": "CHANCELLOR",
    "monarch": "MONARCH",
    "king": "KING",
    "queen": "QUEEN",
    "emperor": "EMPEROR",
    "chief": "CHIEF",
    "secretary": "SECRETARY",
    "minister": "MINISTER",
    "spokesperson": "SPOKESPERSON",
    "owner": "OWNER",
    "parent": "PARENT_ORGANIZATION",
    "subsidiaries": "SUBSIDIARY",
    "location": "LOCATION",
    "venue": "VENUE",
    "country": "COUNTRY",
    "state": "STATE",
    "city": "CITY",
    "county": "COUNTY",
    "region": "REGION",
    "continent": "CONTINENT",
    "battles": "BATTLES",
    "wars": "WARS",
    "awards": "AWARDS",
    "education": "EDUCATION",
    "alma_mater": "ALMA_MATER",
    "occupation": "OCCUPATION",
    "profession": "PROFESSION",
    "party": "POLITICAL_PARTY",
    "nationality": "NATIONALITY",
    "citizenship": "CITIZENSHIP",
    "birth_date": "BIRTH_DATE",
    "birth_place": "BIRTH_PLACE",
    "death_date": "DEATH_DATE",
    "death_place": "DEATH_PLACE",
    "resting_place": "RESTING_PLACE",
    "spouse": "SPOUSE",
    "children": "CHILDREN",
    "parents": "PARENTS",
    "relatives": "RELATIVES",
    "call_sign": "CALL_SIGN",
    "license_plate": "LICENSE_PLATE",
    "airport_code": "AIRPORT_CODE",
    "population": "POPULATION",
    "area": "AREA",
    "density": "DENSITY",
    "elevation": "ELEVATION",
    "website": "WEBSITE",
    "footnotes": "FOOTNOTES",
    "date": "DATE",
    "place": "PLACE",
    "result": "RESULT",
    "territory": "TERRITORIAL_CHANGES",
    "combatants": "COMBATANTS",
    "commander1": "COMMANDER_1",
    "commander2": "COMMANDER_2",
    "strength1": "STRENGTH_1",
    "strength2": "STRENGTH_2",
    "casualties1": "CASUALTIES_1",
    "casualties2": "CASUALTIES_2",
    "partof": "PART_OF",
    "campaignbox": "CAMPAIGNBOX",
}


class InfoboxExtractor:
    def __init__(self, wiki_api_url: str = "https://en.wikipedia.org/w/api.php"):
        self.api_url = wiki_api_url
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "NewsKG/1.0"})

    # ─── NEW: extract directly from provided wikitext ────────────────────
    def extract_from_wikitext(self, article_title: str, wikitext: str) -> List[Tuple[str, str, str]]:
        """
        Extract triples from the given wikitext (no API calls).
        Returns list of (subject, relation, object) triples.
        """
        if not wikitext:
            return []
        infobox_text = self._extract_infobox(wikitext)
        if not infobox_text:
            return []
        triples = []
        for key, value in self._parse_infobox(infobox_text):
            rel = INFOBOX_RELATION_MAP.get(key.lower())
            if rel:
                triples.append((article_title, rel, value))
        return triples

    # ─── Legacy API‑based method (kept for backwards compatibility) ────
    def extract_from_article(self, article_title: str) -> List[Tuple[str, str, str]]:
        """Fetch wikitext via API and extract triples."""
        wikitext = self._fetch_wikitext(article_title)
        if not wikitext:
            log.warning("no_wikitext", article=article_title)
            return []
        return self.extract_from_wikitext(article_title, wikitext)

    def _fetch_wikitext(self, title: str) -> str:
        params = {
            "action": "query",
            "prop": "revisions",
            "rvprop": "content",
            "titles": title,
            "format": "json",
        }
        try:
            resp = self.session.get(self.api_url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            pages = data.get("query", {}).get("pages", {})
            for page in pages.values():
                if "revisions" in page:
                    return page["revisions"][0]["*"]
        except Exception as e:
            log.error("wikitext_fetch_failed", title=title, error=str(e))
        return ""

    # ─── Infobox parsing helpers (unchanged) ──────────────────────────
    def _extract_infobox(self, wikitext: str) -> str:
        start_match = re.search(r"\{\{\s*Infobox", wikitext, re.IGNORECASE)
        if not start_match:
            return ""
        start = start_match.start()
        depth = 0
        i = start
        while i < len(wikitext):
            if wikitext[i:i+2] == "{{":
                depth += 1
                i += 2
            elif wikitext[i:i+2] == "}}":
                depth -= 1
                i += 2
                if depth == 0:
                    return wikitext[start:i]
            else:
                i += 1
        return ""

    def _parse_infobox(self, infobox_text: str) -> List[Tuple[str, str]]:
        infobox_text = re.sub(r"<!--.*?-->", "", infobox_text, flags=re.DOTALL)
        lines = infobox_text.split("\n")
        fields = []
        current_key = None
        current_value = ""

        for line in lines:
            match = re.match(r"^\s*\|\s*([^=]+?)\s*=\s*(.*)$", line)
            if match:
                if current_key:
                    fields.append((current_key.strip(), current_value.strip()))
                current_key = match.group(1).strip()
                current_value = match.group(2).strip()
            else:
                if current_key is not None:
                    current_value += " " + line.strip()

        if current_key:
            fields.append((current_key.strip(), current_value.strip()))

        cleaned = []
        for key, value in fields:
            # Remove wikilinks
            value = re.sub(r"\[\[([^\]|]*?\|)?([^\]|]*?)\]\]", lambda m: m.group(2) or m.group(1) or "", value)
            # Remove templates
            value = re.sub(r"\{\{[^}]+\}\}", "", value)
            # Remove HTML tags
            value = re.sub(r"<[^>]+>", "", value)
            # Clean spaces
            value = re.sub(r"\s+", " ", value).strip()
            if value:
                cleaned.append((key, value))
        return cleaned