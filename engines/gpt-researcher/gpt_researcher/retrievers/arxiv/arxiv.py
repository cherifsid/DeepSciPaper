import os
import random
import re
import time
from threading import Lock

import arxiv


_LAST_ARXIV_REQUEST_AT = 0.0
_ARXIV_REQUEST_LOCK = Lock()


def _normalize_arxiv_query(query: str) -> str:
    """Convert long natural-language prompts into compact arXiv API queries."""
    query = re.sub(r"https?://\S+", " ", query or "")
    query = re.sub(r"site:\S+", " ", query, flags=re.IGNORECASE)
    query = re.sub(r"[“”]", '"', query)
    quoted = [q.strip() for q in re.findall(r'"([^"]{3,80})"', query)]

    lowered = query.lower()
    priority_phrases = []
    phrase_map = [
        ("patent classification", "patent classification"),
        ("sustainable development goals", "sustainable development goals"),
        ("sdg", "SDG"),
        ("weak supervision", "weak supervision"),
        ("large language model", "large language model"),
        ("dataset", "dataset"),
        ("benchmark", "benchmark"),
        ("reproducibility", "reproducibility"),
    ]
    for needle, phrase in phrase_map:
        if needle in lowered and phrase not in priority_phrases:
            priority_phrases.append(phrase)

    for phrase in quoted:
        clean = re.sub(r"[^A-Za-z0-9 -]", " ", phrase).strip()
        if 3 <= len(clean) <= 60 and clean.lower() not in {p.lower() for p in priority_phrases}:
            priority_phrases.append(clean)

    if not priority_phrases:
        words = re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", query)
        stopwords = {
            "search", "primary", "literature", "documents", "available", "online",
            "provide", "relevant", "sources", "links", "references", "materials",
            "studies", "repositories", "scholarly", "publications", "reputable",
            "journals", "focus", "following", "types", "ensure", "includes",
        }
        priority_phrases = [w for w in words if w.lower() not in stopwords][:6]

    if not priority_phrases:
        return "all:patent AND all:classification"

    terms = []
    for phrase in priority_phrases[:5]:
        phrase = phrase.strip()
        if not phrase:
            continue
        if " " in phrase:
            terms.append(f'all:"{phrase}"')
        else:
            terms.append(f"all:{phrase}")
    return " AND ".join(terms[:3]) if terms else "all:patent AND all:classification"


def _alternate_arxiv_queries(api_query: str) -> list[str]:
    """Return progressively looser arXiv queries for sparse or rate-limited searches."""
    queries = [api_query]
    if " AND " in api_query:
        queries.append(api_query.replace(" AND ", " OR "))
    general_queries = [
        'all:"patent classification"',
        'all:patent AND all:SDG',
        'all:patent AND all:"sustainable development"',
        "all:patent AND all:classification",
    ]
    for query in general_queries:
        if query not in queries:
            queries.append(query)
    return queries[:4]


def _is_relevant_to_query(result, original_query: str) -> bool:
    """Keep arXiv fallbacks from drifting into broad but irrelevant topics."""
    query = (original_query or "").lower()
    haystack = f"{getattr(result, 'title', '')} {getattr(result, 'summary', '')}".lower()
    if "patent" in query and "patent" not in haystack:
        return False
    if ("sustainable development goal" in query or "sdg" in query) and not (
        "sustainable development goal" in haystack or "sdg" in haystack or "sdgs" in haystack
    ):
        return False
    return True


class ArxivSearch:
    """
    Arxiv API Retriever
    """
    def __init__(self, query, sort='Relevance', query_domains=None):
        self.arxiv = arxiv
        self.query = query
        self.api_query = _normalize_arxiv_query(query)
        assert sort in ['Relevance', 'SubmittedDate'], "Invalid sort criterion"
        self.sort = arxiv.SortCriterion.SubmittedDate if sort == 'SubmittedDate' else arxiv.SortCriterion.Relevance
        

    def search(self, max_results=5):
        """
        Performs the search
        :param query:
        :param max_results:
        :return:
        """
        global _LAST_ARXIV_REQUEST_AT

        # arXiv API is rate limited; be conservative by default.
        cap = int(os.getenv("ARXIV_MAX_RESULTS", "25"))
        max_results = min(int(max_results or 0) or 25, cap)
        page_size = int(os.getenv("ARXIV_PAGE_SIZE", str(min(25, max_results))))
        delay_seconds = float(os.getenv("ARXIV_DELAY_SECONDS", "4.0"))
        num_retries = int(os.getenv("ARXIV_NUM_RETRIES", "3"))

        client = arxiv.Client(page_size=page_size, delay_seconds=delay_seconds, num_retries=num_retries)

        arxiv_gen = []
        # arXiv's API terms require one connection at a time and at least
        # three seconds between requests across all machines under your control.
        with _ARXIV_REQUEST_LOCK:
            for api_query in _alternate_arxiv_queries(self.api_query):
                attempts = 0
                while True:
                    attempts += 1
                    try:
                        now = time.monotonic()
                        wait_for = (_LAST_ARXIV_REQUEST_AT + delay_seconds) - now
                        if wait_for > 0:
                            time.sleep(wait_for + random.uniform(0.1, 0.4))
                        _LAST_ARXIV_REQUEST_AT = time.monotonic()
                        arxiv_gen = list(
                            client.results(
                                self.arxiv.Search(
                                    query=api_query,
                                    max_results=max_results,
                                    sort_by=self.sort,
                                )
                            )
                        )
                        break
                    except Exception as exc:
                        # Gracefully degrade instead of crashing deep research runs.
                        message = str(exc)
                        transient = any(code in message for code in ("HTTP 429", "HTTP 502", "HTTP 503", "HTTP 504"))
                        if transient and attempts <= num_retries:
                            backoff = delay_seconds * attempts + random.uniform(0.5, 1.8)
                            time.sleep(backoff)
                            continue
                        if transient:
                            print(f"ArxivSearch skipped after transient rate-limit/server response: {exc}")
                            return []
                        print(f"Error with retriever ArxivSearch: {exc}")
                        arxiv_gen = []
                        break
                if arxiv_gen:
                    break

        arxiv_gen = [result for result in arxiv_gen if _is_relevant_to_query(result, self.query)]

        search_result = []

        for result in arxiv_gen:
            entry_id = getattr(result, "entry_id", "") or ""
            abstract_url = entry_id.replace("http://", "https://")
            if not abstract_url and result.pdf_url:
                abstract_url = result.pdf_url.replace("/pdf/", "/abs/").removesuffix(".pdf")
            pdf_url = (result.pdf_url or "").replace("http://", "https://")
            authors = ", ".join(str(author) for author in getattr(result, "authors", [])[:8])
            published = getattr(result, "published", None)
            published_text = published.date().isoformat() if published else ""
            raw_content = (
                f"Title: {result.title}\n"
                f"Authors: {authors}\n"
                f"Published: {published_text}\n"
                f"Abstract URL: {abstract_url}\n"
                f"PDF URL: {pdf_url}\n"
                f"Summary: {result.summary}"
            )

            search_result.append({
                "title": result.title,
                "href": abstract_url or pdf_url,
                "pdf_url": pdf_url,
                "body": result.summary,
                "raw_content": raw_content,
            })
        
        return search_result
