"""Text extraction for binary document formats (PDF, docx, pptx).

local_read can't ``read_text()`` these — PDF is a binary stream and Office
formats are zip containers. This module turns them into plain text so the
agent can search/cite them like any other source.

Design:
  - PDF goes through ``pdfplumber`` (a main dependency). If it's somehow
    absent, callers get a clear degrade message rather than a crash.
  - docx/pptx are OOXML zips; their text lives in ``<w:t>`` / ``<a:t>`` XML
    nodes, so we extract with the standard library alone (zipfile + regex).
    No third-party dependency, nothing to install.

``extract_text`` returns the extracted text, or None when the path isn't an
extractable document (caller falls back to read_text), or a ``DegradeNotice``
when the format is known but the optional library is absent.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

# Formats this module knows how to turn into text. Kept in sync with
# local_common.DOC_EXTS so local_read routes them here instead of skipping.
PDF_EXTS = {".pdf"}
OOXML_EXTS = {".docx", ".pptx"}
DOC_EXTS = PDF_EXTS | OOXML_EXTS

# Page separator inserted between PDF pages so line ranges + citations stay
# meaningful and the agent can tell where a page boundary fell.
_PAGE_SEP = "\n\n--- page {n} ---\n\n"

# OOXML text nodes: Word uses <w:t>, PowerPoint uses <a:t>. The optional
# attributes (e.g. xml:space="preserve") mean we match <w:t ...> too.
_OOXML_TEXT_RE = re.compile(r"<(?:w|a):t[^>]*>(.*?)</(?:w|a):t>", re.S)
# Paragraph / line breaks → newlines so structure survives extraction.
_OOXML_BREAK_RE = re.compile(r"<(?:w:p|a:p|w:br|a:br)\b[^>]*/?>")


class DegradeNotice(str):
    """A user-facing message returned when a known format can't be extracted
    because its optional library is missing. Subclasses str so callers can
    surface it directly, but ``isinstance(x, DegradeNotice)`` distinguishes it
    from real extracted text."""


def is_extractable(path: Path) -> bool:
    return path.suffix.lower() in DOC_EXTS


def extract_text(path: Path) -> str | DegradeNotice | None:
    """Extract plain text from a document.

    Returns:
      - str: the extracted text (possibly empty if the doc has no text layer)
      - DegradeNotice: a known format whose optional library isn't installed
      - None: not an extractable document (caller should try read_text)
    """
    suffix = path.suffix.lower()
    if suffix in PDF_EXTS:
        return _extract_pdf(path)
    if suffix in OOXML_EXTS:
        return _extract_ooxml(path)
    return None


def _extract_pdf(path: Path) -> str | DegradeNotice:
    try:
        import pdfplumber
    except ImportError:
        return DegradeNotice(
            f"'{path.name}' is a PDF, but the pdfplumber library isn't "
            "available. Reinstall SearchClaw, or run: pip install pdfplumber."
        )

    parts: list[str] = []
    empty_pages = 0
    try:
        with pdfplumber.open(str(path)) as pdf:
            for i, page in enumerate(pdf.pages, 1):
                text = page.extract_text() or ""
                if not text.strip():
                    empty_pages += 1
                if i > 1:
                    parts.append(_PAGE_SEP.format(n=i))
                parts.append(text)
    except Exception as e:  # corrupt/encrypted PDF — report, don't crash
        return DegradeNotice(f"Failed to read PDF '{path.name}': {e}")

    body = "".join(parts).strip()
    if empty_pages and not body:
        return DegradeNotice(
            f"'{path.name}' has no extractable text — it's likely a scanned "
            "PDF (images only). OCR is not available."
        )
    if empty_pages:
        body += (
            f"\n\n[Note: {empty_pages} page(s) had no text layer — likely "
            "scanned images, not extracted.]"
        )
    return body


def _extract_ooxml(path: Path) -> str | DegradeNotice:
    """Extract text from a docx/pptx zip via its XML parts (no deps)."""
    suffix = path.suffix.lower()
    try:
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            if suffix == ".docx":
                targets = [n for n in names if n == "word/document.xml"]
            else:  # .pptx — one xml per slide, keep slide order
                targets = sorted(
                    n for n in names
                    if n.startswith("ppt/slides/slide") and n.endswith(".xml")
                )
            chunks: list[str] = []
            for name in targets:
                xml = z.read(name).decode("utf-8", "replace")
                chunks.append(_ooxml_xml_to_text(xml))
    except (zipfile.BadZipFile, KeyError, OSError) as e:
        return DegradeNotice(f"Failed to read '{path.name}': {e}")

    sep = "\n\n--- slide ---\n\n" if suffix == ".pptx" else ""
    return sep.join(c for c in chunks if c).strip()


def _ooxml_xml_to_text(xml: str) -> str:
    """Turn an OOXML body into text: paragraph/line breaks become newlines,
    <w:t>/<a:t> runs become their content, then XML entities are unescaped."""
    # Mark paragraph/break boundaries as newlines before pulling text runs;
    # those markers sit between <w:t> nodes, so we emit a newline whenever the
    # gap between two consecutive runs contains one.
    marked = _OOXML_BREAK_RE.sub("\n", xml)
    out: list[str] = []
    pos = 0
    for m in _OOXML_TEXT_RE.finditer(marked):
        if "\n" in marked[pos:m.start()]:
            out.append("\n")
        out.append(m.group(1))
        pos = m.end()
    return _unescape("".join(out))


def _unescape(s: str) -> str:
    return (
        s.replace("&lt;", "<").replace("&gt;", ">")
        .replace("&quot;", '"').replace("&apos;", "'")
        .replace("&amp;", "&")
    )
