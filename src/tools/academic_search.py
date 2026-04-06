"""
Academic search tool — searches academic databases (Semantic Scholar, arXiv).

Specialized for finding peer-reviewed papers, preprints, and academic
sources. Uses the Semantic Scholar API (free, no key required for
basic usage).

Concurrency-safe: multiple academic searches can run in parallel.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from src.core.tool import Tool, ToolUseContext
from src.core.types import Citation, SourceType, ToolResult, ValidationResult

logger = logging.getLogger(__name__)


class AcademicSearchTool(Tool):
    name = "academic_search"
    description = (
        "Search for academic papers using Semantic Scholar. Returns paper titles, "
        "authors, abstracts, and citation counts. Use this for scientific questions "
        "or when peer-reviewed sources are needed."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query for academic papers.",
            },
            "num_results": {
                "type": "integer",
                "description": "Number of papers to return (default: 5, max: 10).",
                "default": 5,
            },
            "year_range": {
                "type": "string",
                "description": "Optional year range filter, e.g. '2020-2024' or '2023-'.",
            },
            "fields_of_study": {
                "type": "string",
                "description": "Optional: comma-separated fields like 'Computer Science,Medicine'.",
            },
        },
        "required": ["query"],
    }

    is_concurrency_safe = True
    is_read_only = True

    def __init__(
        self,
        default_results: int = 5,
        max_results: int = 10,
        max_result_size_chars: int = 20_000,
        http_timeout: int = 30,
    ):
        self.default_results = default_results
        self.max_results = max_results
        self.max_result_size_chars = max_result_size_chars
        self._client = httpx.AsyncClient(timeout=float(http_timeout))

    def prompt(self) -> str:
        return (
            "Use academic_search for scientific or research questions. It searches "
            "Semantic Scholar for peer-reviewed papers. Tips:\n"
            "- Use technical/scientific terminology in your query\n"
            "- Filter by year range for recent research\n"
            "- Check citation counts to gauge paper importance\n"
            "- Papers with high citation counts are generally more established"
        )

    def validate_input(self, args: dict) -> ValidationResult:
        query = args.get("query", "")
        if not query or len(query.strip()) < 2:
            return ValidationResult(valid=False, message="Query must be at least 2 characters")
        return ValidationResult(valid=True)

    async def call(self, args: dict, context: ToolUseContext) -> ToolResult:
        query = args["query"]
        num_results = min(args.get("num_results", self.default_results), self.max_results)
        year_range = args.get("year_range", "")
        fields = args.get("fields_of_study", "")

        try:
            papers = await self._search_semantic_scholar(
                query, num_results, year_range, fields
            )
        except Exception as e:
            logger.error(f"Academic search failed: {e}")
            return ToolResult(
                data=f"Academic search failed: {str(e)}. Try web_search as a fallback.",
                is_error=True,
            )

        if not papers:
            return ToolResult(
                data="No academic papers found for this query. Try different keywords or web_search.",
            )

        # Format results
        formatted_parts = [f"## Academic Search Results: {query}\n"]
        citations = []

        for i, paper in enumerate(papers, 1):
            title = paper.get("title", "Untitled")
            paper_id = paper.get("paperId", "")
            year = paper.get("year", "N/A")
            citation_count = paper.get("citationCount", 0)
            abstract = paper.get("abstract", "No abstract available")
            url = paper.get("url", f"https://www.semanticscholar.org/paper/{paper_id}")

            # Authors
            authors = paper.get("authors", [])
            author_str = ", ".join(a.get("name", "") for a in authors[:5])
            if len(authors) > 5:
                author_str += f" et al. ({len(authors)} total)"

            # Venue
            venue = paper.get("venue", "") or paper.get("journal", {}).get("name", "")

            formatted_parts.append(
                f"### {i}. {title}\n"
                f"**Authors**: {author_str}\n"
                f"**Year**: {year} | **Citations**: {citation_count}\n"
                + (f"**Venue**: {venue}\n" if venue else "")
                + f"**URL**: {url}\n"
                f"**Abstract**: {abstract[:500]}{'...' if abstract and len(abstract) > 500 else ''}\n"
            )

            citations.append(Citation(
                url=url,
                title=title,
                snippet=abstract[:300] if abstract else "",
                source_type=SourceType.ACADEMIC,
            ))

        formatted = "\n".join(formatted_parts)
        return ToolResult(data=formatted, citations=citations)

    async def _search_semantic_scholar(
        self,
        query: str,
        num_results: int,
        year_range: str = "",
        fields: str = "",
    ) -> list[dict]:
        """Search using Semantic Scholar API (free, no key required)."""
        params: dict[str, Any] = {
            "query": query,
            "limit": num_results,
            "fields": "title,authors,abstract,year,citationCount,url,venue,journal,paperId",
        }

        if year_range:
            params["year"] = year_range

        if fields:
            params["fieldsOfStudy"] = fields

        response = await self._client.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params=params,
            headers={"Accept": "application/json"},
        )
        response.raise_for_status()
        data = response.json()

        return data.get("data", [])
