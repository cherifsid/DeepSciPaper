"""Tavily API search retriever for GPT Researcher.

This module provides the TavilySearch class for performing web searches
using the Tavily API.
"""

import json
import os
import re
from typing import Literal, Optional, Sequence

import requests


def _normalize_tavily_query(query: str, max_length: int = 390) -> str:
    """Keep Tavily queries inside its 400-character API limit without losing intent."""
    clean = re.sub(r"\s+", " ", str(query or "")).strip()
    if len(clean) <= max_length:
        return clean

    clean = re.sub(r"https?://\S+", " ", clean)
    clean = re.sub(r"site:\S+", " ", clean, flags=re.IGNORECASE)
    clean = clean.replace("“", '"').replace("”", '"')
    quoted = [item.strip() for item in re.findall(r'"([^"]{3,90})"', clean)]
    lowered = clean.lower()
    priority = []
    for phrase in (
        "patent classification",
        "sustainable development goals",
        "patent SDG mapping",
        "SDG dataset",
        "benchmark",
        "reproducibility",
        "github",
        "open access pdf",
        "peer reviewed",
    ):
        if phrase.lower() in lowered:
            priority.append(phrase)
    words = re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", clean)
    stopwords = {
        "search", "primary", "literature", "documents", "available", "online",
        "provide", "relevant", "sources", "links", "references", "materials",
        "studies", "repositories", "scholarly", "publications", "reputable",
        "journals", "focus", "following", "types", "ensure", "includes",
    }
    selected = []
    for item in quoted + priority + [word for word in words if word.lower() not in stopwords]:
        item = re.sub(r"[^A-Za-z0-9 .:/_-]", " ", item).strip()
        if item and item.lower() not in {existing.lower() for existing in selected}:
            selected.append(item)
        candidate = " ".join(selected)
        if len(candidate) >= max_length:
            break
    return " ".join(selected)[:max_length].rsplit(" ", 1)[0].strip() or clean[:max_length]


class TavilySearch:
    """
    Tavily API Retriever
    """

    def __init__(self, query, headers=None, topic="general", query_domains=None):
        """
        Initializes the TavilySearch object.

        Args:
            query (str): The search query string.
            headers (dict, optional): Additional headers to include in the request. Defaults to None.
            topic (str, optional): The topic for the search. Defaults to "general".
            query_domains (list, optional): List of domains to include in the search. Defaults to None.
        """
        self.query = query
        self.headers = headers or {}
        self.topic = topic
        self.base_url = "https://api.tavily.com/search"
        self.api_key = self.get_api_key()
        self.headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"
        self.query_domains = query_domains or None

    def get_api_key(self):
        """
        Gets the Tavily API key
        Returns:

        """
        api_key = self.headers.get("tavily_api_key")
        if not api_key:
            try:
                api_key = os.environ["TAVILY_API_KEY"]
            except KeyError:
                print(
                    "Tavily API key not found, set to blank. If you need a retriver, please set the TAVILY_API_KEY environment variable."
                )
                return ""
        return api_key


    def _search(
        self,
        query: str,
        search_depth: Literal["basic", "advanced"] = "basic",
        topic: str = "general",
        days: int = 2,
        max_results: int = 10,
        include_domains: Sequence[str] = None,
        exclude_domains: Sequence[str] = None,
        include_answer: bool = False,
        include_raw_content: bool = False,
        include_images: bool = False,
        use_cache: bool = True,
    ) -> dict:
        """
        Internal search method to send the request to the API.
        """

        if not self.api_key:
            return {}

        safe_max_results = max(1, min(int(max_results or 10), int(os.getenv("TAVILY_MAX_RESULTS", "20"))))
        data = {
            "query": _normalize_tavily_query(query),
            "search_depth": search_depth,
            "topic": topic,
            "include_answer": include_answer,
            "include_raw_content": include_raw_content,
            "max_results": safe_max_results,
            "include_images": include_images,
            "use_cache": use_cache,
        }
        if topic == "news":
            data["days"] = days
        if include_domains:
            data["include_domains"] = list(include_domains)
        if exclude_domains:
            data["exclude_domains"] = list(exclude_domains)

        response = requests.post(
            self.base_url, json=data, headers=self.headers, timeout=100
        )

        if response.status_code == 200:
            return response.json()
        else:
            if response.status_code == 400:
                print(f"Tavily bad request: {response.text[:500]}")
            # Raises a HTTPError if the HTTP request returned an unsuccessful status code
            response.raise_for_status()

    def search(self, max_results=10):
        """
        Searches the query
        Returns:

        """
        try:
            # Search the query
            results = self._search(
                self.query,
                search_depth="basic",
                max_results=max_results,
                topic=self.topic,
                include_domains=self.query_domains,
            )
            sources = results.get("results", [])
            if not sources:
                raise Exception("No results found with Tavily API search.")
            # Return the results
            search_response = [
                {"href": obj["url"], "body": obj["content"]} for obj in sources
            ]
        except Exception as e:
            print(f"Error: {e}. Failed fetching sources. Resulting in empty response.")
            search_response = []
        return search_response
