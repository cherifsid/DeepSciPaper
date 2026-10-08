"""Validated search plans shared by research report types."""
import json
import re


def planner_prompt(today: str) -> str:
    return f"""Design a scientific literature search from the user's full brief.
Today is {today}. Return ONLY JSON with a queries list of objects containing
query, researchGoal, scope (direct or supporting), and facet.
Generate 6-10 complementary searches, covering the requested report sections.
Queries must be at most 240 characters, normally 4-14 words, with 2-3 concepts
and at most two AND operators. Do not require all user constraints in every
query. Avoid nested Boolean expressions, arbitrary year lists and publisher
names. Respect explicit dates and exclusions, but retain foundational work.
Begin with direct task searches; add separate architecture, rule/constraint,
feature, dataset and evaluation searches. At least two searches must be direct
and at least four distinct facets must be covered. Broaden through supporting
method searches when the exact intersection is sparse, clearly labeling them.
Each goal must state exactly what evidence to extract and its relevance to
the user's requested outputs. No generic goals or invented findings. Supporting
goals must explicitly require checking applicability to the user's task.
Preserve the actual objects being compared: goods/services descriptions are
not trademark brand names. Do not substitute a neighboring task. Requested
labels and mechanisms are requirements to verify, not proven literature facts.
Do not invent papers, metrics, datasets or claims that an exact solution exists.
"""


def parse_search_plan(raw: str) -> list[dict[str, str]]:
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip()).strip()
    data = json.loads(raw)
    items = data.get("queries") if isinstance(data, dict) else data
    if not isinstance(items, list) or not 6 <= len(items) <= 10:
        raise ValueError("Return 6-10 complementary searches.")
    result, seen, facets = [], set(), set()
    direct = 0
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Each query requires a goal, scope and facet.")
        query = str(item.get("query", "")).strip()
        goal = str(item.get("researchGoal", "")).strip()
        scope = item.get("scope")
        facet = str(item.get("facet", "")).strip().lower()
        if not query or len(query) > 240 or len(re.findall(r"\bAND\b", query)) > 2:
            raise ValueError("Use concise queries under 240 characters with at most two AND operators.")
        if len(goal.split()) < 8 or "high-signal evidence" in goal.lower():
            raise ValueError("Goals must specify concrete evidence and relevance to the task.")
        if scope not in ("direct", "supporting") or not facet:
            raise ValueError("Every search requires a direct/supporting scope and facet.")
        key = " ".join(query.casefold().split())
        if key in seen:
            raise ValueError("Do not repeat queries.")
        seen.add(key)
        facets.add(facet)
        direct += scope == "direct"
        result.append({"query": query, "researchGoal": f"[{scope.capitalize()} / {facet}] {goal}"})
    if direct < 2 or len(facets) < 4:
        raise ValueError("Cover at least four facets and two direct searches.")
    return result
