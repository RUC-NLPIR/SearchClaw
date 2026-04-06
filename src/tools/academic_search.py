"""
Academic search tool — searches academic databases (Semantic Scholar, DBLP, arXiv).

Specialized for finding peer-reviewed papers, preprints, and academic
sources. Supports three backends:
  - Semantic Scholar: citation counts, abstracts, broad coverage (free, no key)
  - DBLP: authoritative CS bibliography, precise metadata (free, no key)
  - arXiv: preprints with full abstracts (free, no key)

Concurrency-safe: multiple academic searches can run in parallel.
"""

from __future__ import annotations

import asyncio
import logging
import re
import xml.etree.ElementTree as ET
from typing import Any
import httpx

from src.core.tool import Tool, ToolUseContext
from src.core.types import Citation, SourceType, ToolResult, ValidationResult

logger = logging.getLogger(__name__)

# XML namespaces for arXiv Atom feed
_ATOM_NS = "http://www.w3.org/2005/Atom"
_ARXIV_NS = "http://arxiv.org/schemas/atom"


class AcademicSearchTool(Tool):
    name = "academic_search"
    description = (
        "Search for academic papers using Semantic Scholar, DBLP, and arXiv. "
        "Returns paper titles, authors, abstracts, citation counts, and links. "
        "Use this for scientific questions or when peer-reviewed sources are needed. "
        "Defaults to searching all three databases in parallel."
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
                "description": "Number of papers to return per source (default: 5, max: 10).",
                "default": 5,
            },
            "year_range": {
                "type": "string",
                "description": "Optional year range filter, e.g. '2020-2024' or '2023-'.",
            },
            "fields_of_study": {
                "type": "string",
                "description": "Optional: comma-separated fields like 'Computer Science,Medicine'. Only applies to Semantic Scholar.",
            },
            "source": {
                "type": "string",
                "description": "Which database to search: 'all' (default), 'semantic_scholar', 'dblp', or 'arxiv'.",
                "enum": ["all", "semantic_scholar", "dblp", "arxiv"],
                "default": "all",
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
        max_result_size_chars: int = 20000,
        http_timeout: int = 30,
    ):
        self.default_results = default_results
        self.max_results = max_results
        self.max_result_size_chars = max_result_size_chars
        self._client = httpx.AsyncClient(timeout=float(http_timeout))

    def prompt(self) -> str:
        return (
            "Use academic_search for scientific or research questions. It searches "
            "Semantic Scholar, DBLP, and arXiv for peer-reviewed papers and preprints. Tips:\n"
            "- Use technical/scientific terminology in your query\n"
            "- Filter by year range for recent research (e.g. '2023-2024')\n"
            "- Use source='semantic_scholar' for citation counts and broad coverage\n"
            "- Use source='dblp' for precise CS bibliographic data (venues, DOIs)\n"
            "- Use source='arxiv' for preprints with full abstracts\n"
            "- Default source='all' searches all three in parallel and merges results"
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
        source = args.get("source", "all")

        papers: list[dict] = []

        if source == "all":
            # Search all three in parallel
            ss_task = self._search_semantic_scholar(query, num_results, year_range, fields)
            dblp_task = self._search_dblp(query, num_results, year_range)
            arxiv_task = self._search_arxiv(query, num_results, year_range)
            results = await asyncio.gather(
                ss_task, dblp_task, arxiv_task, return_exceptions=True
            )
            labels = ["Semantic Scholar", "DBLP", "arXiv"]
            for label, result in zip(labels, results):
                if isinstance(result, Exception):
                    logger.warning(f"{label} search failed: {result}")
                else:
                    papers.extend(result)
            # Deduplicate across sources
            papers = self._deduplicate(papers)
        elif source == "semantic_scholar":
            try:
                papers = await self._search_semantic_scholar(query, num_results, year_range, fields)
            except Exception as e:
                logger.error(f"Semantic Scholar search failed: {e}")
                return ToolResult(data=f"Semantic Scholar search failed: {e}", is_error=True)
        elif source == "dblp":
            try:
                papers = await self._search_dblp(query, num_results, year_range)
            except Exception as e:
                logger.error(f"DBLP search failed: {e}")
                return ToolResult(data=f"DBLP search failed: {e}", is_error=True)
        elif source == "arxiv":
            try:
                papers = await self._search_arxiv(query, num_results, year_range)
            except Exception as e:
                logger.error(f"arXiv search failed: {e}")
                return ToolResult(data=f"arXiv search failed: {e}", is_error=True)
        else:
            return ToolResult(data=f"Unknown source: {source}", is_error=True)

        if not papers:
            return ToolResult(
                data="No academic papers found for this query. Try different keywords or web_search.",
            )

        # Format results
        source_label = "All Databases" if source == "all" else source.replace("_", " ").title()
        formatted_parts = [f"## Academic Search Results ({source_label}): {query}\n"]
        citations = []

        for i, paper in enumerate(papers, 1):
            title = paper.get("title", "Untitled")
            year = paper.get("year", "N/A")
            citation_count = paper.get("citationCount")
            abstract = paper.get("abstract", "") or ""
            url = paper.get("url", "")
            paper_source = paper.get("_source", "")

            # Authors
            authors = paper.get("authors", [])
            author_str = ", ".join(a.get("name", "") for a in authors[:5])
            if len(authors) > 5:
                author_str += f" et al. ({len(authors)} total)"

            # Venue
            venue = paper.get("venue", "")

            parts = [f"### {i}. {title}\n"]
            if author_str:
                parts.append(f"**Authors**: {author_str}\n")
            year_cite = f"**Year**: {year}"
            if citation_count is not None:
                year_cite += f" | **Citations**: {citation_count}"
            parts.append(year_cite + "\n")
            if venue:
                parts.append(f"**Venue**: {venue}\n")
            if paper_source:
                parts.append(f"**Source**: {paper_source}\n")
            if url:
                parts.append(f"**URL**: {url}\n")
            if abstract:
                parts.append(
                    f"**Abstract**: {abstract[:500]}{'...' if len(abstract) > 500 else ''}\n"
                )

            formatted_parts.append("".join(parts))

            if url:
                citations.append(Citation(
                    url=url,
                    title=title,
                    snippet=abstract[:300] if abstract else "",
                    source_type=SourceType.ACADEMIC,
                ))

        formatted = "\n".join(formatted_parts)
        return ToolResult(data=formatted, citations=citations)

    # ── Semantic Scholar ─────────────────────────────────────────────

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

        papers = data.get("data", [])
        # Normalize: add _source tag, ensure venue from journal fallback
        for p in papers:
            p["_source"] = "Semantic Scholar"
            if not p.get("venue"):
                journal = p.get("journal")
                if isinstance(journal, dict):
                    p["venue"] = journal.get("name", "")
        return papers

    # ── DBLP ─────────────────────────────────────────────────────────

    async def _search_dblp(
        self,
        query: str,
        num_results: int,
        year_range: str = "",
    ) -> list[dict]:
        """Search using DBLP API (free, no key required)."""
        params: dict[str, Any] = {
            "q": query,
            "format": "json",
            "h": num_results,
            "c": 0,  # no auto-completions
        }

        response = await self._client.get(
            "https://dblp.org/search/publ/api",
            params=params,
            headers={"Accept": "application/json"},
        )
        response.raise_for_status()
        data = response.json()

        hits = data.get("result", {}).get("hits", {}).get("hit", [])
        if not hits:
            return []

        # Parse year range for client-side filtering
        year_start, year_end = self._parse_year_range(year_range)

        papers = []
        for hit in hits:
            info = hit.get("info", {})
            year_str = info.get("year", "")
            year = int(year_str) if year_str.isdigit() else None

            # Client-side year filtering (DBLP API doesn't support it)
            if year is not None:
                if year_start and year < year_start:
                    continue
                if year_end and year > year_end:
                    continue

            # Extract authors — can be a list or a single dict
            raw_authors = info.get("authors", {}).get("author", [])
            if isinstance(raw_authors, dict):
                raw_authors = [raw_authors]
            authors = [{"name": a.get("text", a) if isinstance(a, dict) else str(a)} for a in raw_authors]

            # Prefer electronic edition URL, fall back to dblp record URL
            url = info.get("ee", "") or info.get("url", "")
            # ee can be a list (multiple editions) — take the first
            if isinstance(url, list):
                url = url[0] if url else ""

            papers.append({
                "title": info.get("title", "Untitled"),
                "authors": authors,
                "abstract": "",  # DBLP doesn't provide abstracts
                "year": year or "N/A",
                "citationCount": None,  # DBLP doesn't provide citation counts
                "url": url,
                "venue": info.get("venue", ""),
                "paperId": info.get("key", ""),
                "_source": "DBLP",
            })

        return papers

    # ── arXiv ────────────────────────────────────────────────────────

    async def _search_arxiv(
        self,
        query: str,
        num_results: int,
        year_range: str = "",
    ) -> list[dict]:
        """Search using arXiv API (free, no key required)."""
        # Build search query (httpx handles URL encoding, so use plain text)
        search_query = f"all:{query}"

        # Add date filter if year range specified
        if year_range:
            year_start, year_end = self._parse_year_range(year_range)
            if year_start or year_end:
                start_date = f"{year_start or 1991}01010000"
                end_date = f"{year_end or 2099}12312359"
                search_query += f" AND submittedDate:[{start_date} TO {end_date}]"

        params: dict[str, Any] = {
            "search_query": search_query,
            "start": 0,
            "max_results": num_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }

        response = await self._client.get(
            "https://export.arxiv.org/api/query",
            params=params,
        )
        response.raise_for_status()

        # Parse Atom XML
        root = ET.fromstring(response.text)

        papers = []
        for entry in root.findall(f"{{{_ATOM_NS}}}entry"):
            title = entry.findtext(f"{{{_ATOM_NS}}}title", "Untitled")
            # Clean up title whitespace (arXiv titles often have newlines)
            title = re.sub(r"\s+", " ", title).strip()

            # Authors
            authors = []
            for author_el in entry.findall(f"{{{_ATOM_NS}}}author"):
                name = author_el.findtext(f"{{{_ATOM_NS}}}name", "")
                if name:
                    authors.append({"name": name})

            # Abstract
            abstract = entry.findtext(f"{{{_ATOM_NS}}}summary", "") or ""
            abstract = re.sub(r"\s+", " ", abstract).strip()

            # URL (the <id> element is the abstract page URL)
            url = entry.findtext(f"{{{_ATOM_NS}}}id", "") or ""

            # Published date → year
            published = entry.findtext(f"{{{_ATOM_NS}}}published", "")
            year: int | str = "N/A"
            if published and len(published) >= 4:
                try:
                    year = int(published[:4])
                except ValueError:
                    pass

            # Primary category as venue
            primary_cat = entry.find(f"{{{_ARXIV_NS}}}primary_category")
            venue = ""
            if primary_cat is not None:
                venue = primary_cat.get("term", "")

            # arXiv ID from URL
            paper_id = url.split("/abs/")[-1] if "/abs/" in url else ""

            papers.append({
                "title": title,
                "authors": authors,
                "abstract": abstract,
                "year": year,
                "citationCount": None,  # arXiv doesn't provide citation counts
                "url": url,
                "venue": f"arXiv:{venue}" if venue else "arXiv",
                "paperId": paper_id,
                "_source": "arXiv",
            })

        return papers

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _parse_year_range(year_range: str) -> tuple[int | None, int | None]:
        """Parse a year range string like '2020-2024' or '2023-' into (start, end)."""
        if not year_range:
            return None, None
        parts = year_range.split("-")
        start = int(parts[0]) if parts[0].strip().isdigit() else None
        end = int(parts[1]) if len(parts) > 1 and parts[1].strip().isdigit() else None
        return start, end

    @staticmethod
    def _deduplicate(papers: list[dict]) -> list[dict]:
        """Deduplicate papers by normalized title, keeping the most informative version."""
        seen: dict[str, dict] = {}
        for paper in papers:
            key = re.sub(r"\s+", " ", paper.get("title", "").lower().strip())
            if not key:
                continue
            if key in seen:
                existing = seen[key]
                # Prefer the version with an abstract
                if not existing.get("abstract") and paper.get("abstract"):
                    seen[key] = paper
                # If both have abstracts, prefer the one with citation count
                elif (existing.get("abstract") and paper.get("abstract")
                      and existing.get("citationCount") is None
                      and paper.get("citationCount") is not None):
                    seen[key] = paper
            else:
                seen[key] = paper
        return list(seen.values())
