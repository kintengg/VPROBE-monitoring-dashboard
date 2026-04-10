from __future__ import annotations

import json
import os
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

GEMINI_SEARCH_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
MAX_RETURNED_MATCHES = 5
SUPPORTED_QUERY_REGIONS = ("head region", "upper clothing", "lower clothing")
SUPPORTED_QUERY_COLORS = ("black", "blue", "brown", "gray", "green", "orange", "pink", "purple", "red", "white", "yellow")
SEARCH_QUERY_PARSE_TIMEOUT_SECONDS = 6
SEARCH_RESULT_RANK_TIMEOUT_SECONDS = 8


def _response_text(payload: dict[str, Any]) -> str:
    candidates = payload.get("candidates") or []
    if not candidates:
        return ""

    content = candidates[0].get("content") or {}
    parts = content.get("parts") or []
    text_parts = [str(part.get("text") or "") for part in parts if isinstance(part, dict)]
    return "\n".join(part for part in text_parts if part).strip()


def _call_gemini_json(prompt: dict[str, Any], *, timeout: int = 20) -> Any:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None

    request = urllib_request.Request(
        f"{GEMINI_SEARCH_URL}?key={api_key}",
        data=json.dumps(
            {
                "contents": [{"parts": [{"text": json.dumps(prompt)}]}],
                "generationConfig": {"temperature": 0.1, "responseMimeType": "application/json"},
            }
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib_request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib_error.HTTPError, urllib_error.URLError, TimeoutError, json.JSONDecodeError):
        return None

    response_text = _response_text(payload)
    if not response_text:
        return None

    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        return None


def parse_search_query(query: str, locations: list[dict[str, Any]]) -> dict[str, Any]:
    if not query.strip() or not locations:
        return {}

    prompt = {
        "instructions": (
            "You convert a natural-language surveillance search query into structured JSON. "
            "Use the provided campus locations to resolve place mentions like Xavier Hall or Kostka Hall to the closest matching location id when possible. "
            "Only mark attributes as strict searchable terms when they are likely visible in available pedestrian metadata, such as colors, body regions, garments, accessories, logos, OCR text, and explicit location references from vision-enriched thumbnails. "
            "Treat traits like short/tall, flowy, sleeveless, formal, pace, or inferred intent as softTerms unless a color-body-region mapping is still useful. "
            "If a sentence implies a colored body region, populate regionColorRequirements using only the supported regions. "
            "Return strict JSON with shape {\"locationId\": str|null, \"locationName\": str|null, \"appearanceTerms\": [str], \"softTerms\": [str], \"unsupportedTerms\": [str], \"regionColorRequirements\": [{\"region\": str, \"colors\": [str]}], \"summary\": str}."
        ),
        "query": query.strip(),
        "supportedRegions": list(SUPPORTED_QUERY_REGIONS),
        "supportedColors": list(SUPPORTED_QUERY_COLORS),
        "locations": [
            {
                "id": str(location.get("id") or ""),
                "name": str(location.get("name") or ""),
                "address": str(location.get("address") or ""),
                "description": str(location.get("description") or ""),
            }
            for location in locations
        ],
    }

    parsed = _call_gemini_json(prompt, timeout=SEARCH_QUERY_PARSE_TIMEOUT_SECONDS)
    if not isinstance(parsed, dict):
        return {}

    location_map = {str(location.get("id") or ""): location for location in locations}
    location_id = str(parsed.get("locationId") or "").strip()
    if location_id not in location_map:
        location_id = ""

    def _normalize_terms(value: Any, *, limit: int = 12) -> list[str]:
        if not isinstance(value, list):
            return []

        normalized: list[str] = []
        seen: set[str] = set()
        for item in value:
            term = str(item or "").strip().lower()
            if not term or term in seen:
                continue
            seen.add(term)
            normalized.append(term)
            if len(normalized) >= limit:
                break
        return normalized

    region_requirements: list[dict[str, Any]] = []
    raw_requirements = parsed.get("regionColorRequirements")
    if isinstance(raw_requirements, list):
        seen_requirements: set[tuple[str, tuple[str, ...]]] = set()
        for item in raw_requirements:
            if not isinstance(item, dict):
                continue
            region = str(item.get("region") or "").strip().lower()
            if region not in SUPPORTED_QUERY_REGIONS:
                continue
            colors = [color for color in _normalize_terms(item.get("colors"), limit=5) if color in SUPPORTED_QUERY_COLORS]
            if not colors:
                continue
            key = (region, tuple(colors))
            if key in seen_requirements:
                continue
            seen_requirements.add(key)
            region_requirements.append({"region": region, "colors": colors})

    location_name = str(parsed.get("locationName") or "").strip()
    if location_id:
        location_name = str(location_map[location_id].get("name") or location_name)

    summary = str(parsed.get("summary") or "").strip()
    return {
        "locationId": location_id or None,
        "locationName": location_name or None,
        "appearanceTerms": _normalize_terms(parsed.get("appearanceTerms")),
        "softTerms": _normalize_terms(parsed.get("softTerms")),
        "unsupportedTerms": _normalize_terms(parsed.get("unsupportedTerms")),
        "regionColorRequirements": region_requirements,
        "summary": summary,
    }


def rank_pedestrian_matches(query: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not query.strip() or not candidates:
        return []

    prompt = {
        "instructions": (
            "You are ranking deduplicated pedestrian track summaries from surveillance footage. "
            "Each candidate is one tracked pedestrian summarized from representative bounding-box crops. "
            "Candidates may also include Cloud Vision labels, detected objects, logos, and visible OCR text from the representative thumbnail; use that as additional evidence. "
            "Treat approximate appearance matches as useful: maroon/burgundy/wine can match red, purple, or pink tones; white and gray can be confused by dim lighting or camera exposure; "
            "tops/shirts/blouses map to upper clothing; hats/caps map to the head region; shorts/pants/skirts map to lower clothing. "
            "If the query specifies a body region such as hat/head, top/upper clothing, or shorts/lower clothing, do not treat a color on a different body region as a match. "
            "Do not require exact wording. Prefer candidates whose color family and body region align with the query, "
            "and tolerate missing details like sleeve length or fabric shape when those details are not present in the summaries. "
            "Return strict JSON with shape {\"matches\": [{\"id\": str, \"confidence\": int, \"reason\": str}]}. "
            "Only include plausible matches for the query. Confidence must be 0-100. "
            f"Return at most {MAX_RETURNED_MATCHES} matches, ordered best-first."
        ),
        "query": query.strip(),
        "candidates": candidates,
    }

    parsed = _call_gemini_json(prompt, timeout=SEARCH_RESULT_RANK_TIMEOUT_SECONDS)
    if parsed is None:
        return []

    raw_matches = parsed.get("matches") if isinstance(parsed, dict) else parsed
    if not isinstance(raw_matches, list):
        return []

    ranked_matches: list[dict[str, Any]] = []
    for item in raw_matches:
        if not isinstance(item, dict):
            continue

        candidate_id = str(item.get("id") or "").strip()
        if not candidate_id:
            continue

        try:
            confidence = int(round(float(item.get("confidence", 0))))
        except (TypeError, ValueError):
            confidence = 0

        reason = str(item.get("reason") or "Potential visual match based on the representative pedestrian track.").strip()
        ranked_matches.append(
            {
                "id": candidate_id,
                "confidence": max(0, min(100, confidence)),
                "reason": reason,
            }
        )

    return ranked_matches[:MAX_RETURNED_MATCHES]