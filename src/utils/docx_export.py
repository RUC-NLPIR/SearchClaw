"""DOCX export helpers for final Markdown reports."""

from __future__ import annotations

import re
import tempfile
import zipfile
from io import BytesIO
from pathlib import Path
from xml.etree import ElementTree as ET


DOCX_MEDIA_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
EN_FONT = "Times New Roman"
ET.register_namespace("w", W_NS)


def markdown_to_docx_bytes(
    markdown: str,
    *,
    reference_docx: str | Path | None = None,
) -> bytes:
    """
    Convert Markdown to DOCX using pypandoc.

    Pandoc gives us much better Markdown coverage than hand-writing a Word
    document with python-docx, especially for headings, lists, tables and code
    blocks.  The reference_docx hook lets us later plug in a school-styled Word
    template without changing the API.
    """
    try:
        import pypandoc
    except ImportError as exc:
        raise RuntimeError("pypandoc is not installed; cannot export DOCX.") from exc

    source = normalize_markdown_for_docx(markdown.strip())
    if not source:
        raise ValueError("Nothing to export.")

    extra_args = ["--standalone"]
    if reference_docx:
        ref_path = Path(reference_docx)
        if ref_path.exists():
            extra_args.append(f"--reference-doc={ref_path}")

    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        pypandoc.convert_text(
            source,
            "docx",
            format="gfm",
            outputfile=str(tmp_path),
            extra_args=extra_args,
        )
        return apply_docx_chinese_styles(tmp_path.read_bytes())
    except OSError as exc:
        raise RuntimeError("Pandoc is unavailable; ensure pandoc or pypandoc_binary is installed on the server.") from exc
    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def normalize_markdown_for_docx(markdown: str) -> str:
    """Normalize model-generated Markdown before sending it to Pandoc."""
    source = normalize_markdown_blockquotes(markdown)
    source = normalize_markdown_horizontal_rules(source)
    return normalize_markdown_tables(source)


def normalize_markdown_blockquotes(markdown: str) -> str:
    """
    Convert Markdown blockquotes to normal paragraphs for Word export.

    The report often starts with short quoted summary lines. Pandoc maps these
    to Word's BlockText style, which adds left indentation and may collapse
    line breaks.  For this project the quote styling is not meaningful, so we
    strip the quote markers and add Markdown hard breaks between consecutive
    quoted lines.
    """
    lines = markdown.splitlines()
    out: list[str] = []
    in_fence = False

    for idx, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            out.append(line)
            continue

        if in_fence or not _is_blockquote_line(line):
            out.append(line)
            continue

        content = _strip_blockquote_marker(line).rstrip()
        next_line = lines[idx + 1] if idx + 1 < len(lines) else ""
        if content and _is_blockquote_line(next_line) and _strip_blockquote_marker(next_line).strip():
            content += "  "
        out.append(content)

    return "\n".join(out)


def normalize_markdown_horizontal_rules(markdown: str) -> str:
    """Remove standalone Markdown horizontal rules from DOCX exports."""
    lines = markdown.splitlines()
    out: list[str] = []
    in_fence = False

    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            out.append(line)
            continue

        if not in_fence and _is_horizontal_rule(line):
            continue

        out.append(line)

    return "\n".join(out)


def normalize_markdown_tables(markdown: str) -> str:
    """
    Make loose model-generated pipe tables acceptable to Pandoc.

    Models sometimes output a header row such as `| 来源 | 内容 |` without the
    required separator row. Pandoc/GFM needs `| --- | --- |`, so insert it when
    a pipe row with 2+ cells is not followed by a separator.
    """
    lines = markdown.splitlines()
    out: list[str] = []
    in_fence = False
    idx = 0

    while idx < len(lines):
        line = lines[idx]
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            out.append(line)
            idx += 1
            continue

        if in_fence or not _is_pipe_row(stripped):
            out.append(line)
            idx += 1
            continue

        table_lines: list[str] = []
        while idx < len(lines) and _is_pipe_row(lines[idx].strip()):
            table_lines.append(lines[idx])
            idx += 1

        out.extend(_normalize_table_block(table_lines))

    return "\n".join(out)


def apply_docx_chinese_styles(docx_bytes: bytes) -> bytes:
    """
    Patch generated DOCX so headings use 黑体 and normal content uses 宋体.

    Pandoc's default reference doc is generic; setting both document defaults
    and concrete paragraph runs gives predictable Chinese rendering without
    requiring a user-provided reference.docx.
    """
    source = BytesIO(docx_bytes)
    target = BytesIO()
    with zipfile.ZipFile(source, "r") as zin, zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "word/styles.xml":
                data = _patch_styles_xml(data)
            elif item.filename == "word/document.xml":
                data = _patch_document_xml(data)
            zout.writestr(item, data)
    return target.getvalue()


def safe_docx_filename(title: str | None) -> str:
    stem = (title or "research-report").strip() or "research-report"
    stem = re.sub(r'[\\/:*?"<>|\r\n\t]+', "-", stem)
    stem = re.sub(r"\s+", " ", stem).strip(" .-")
    if not stem:
        stem = "research-report"
    return stem[:80] + ".docx"


def _normalize_table_block(table_lines: list[str]) -> list[str]:
    if len(table_lines) < 2:
        return table_lines

    first = table_lines[0].strip()
    second = table_lines[1].strip()
    if _is_separator_row(first):
        return table_lines
    if _is_separator_row(second):
        return table_lines

    cell_count = len(_pipe_cells(first))
    return [
        table_lines[0],
        "| " + " | ".join("---" for _ in range(cell_count)) + " |",
        *table_lines[1:],
    ]


def _patch_styles_xml(data: bytes) -> bytes:
    root = ET.fromstring(data)
    _patch_doc_defaults(root)
    _force_existing_colors_black(root)
    for style in root.findall(_w("style")):
        style_id = style.attrib.get(_w("styleId"), "")
        name_el = style.find(_w("name"))
        style_name = name_el.attrib.get(_w("val"), "") if name_el is not None else ""
        font = "黑体" if _is_heading_style(style_id, style_name) else "宋体"
        rpr = _ensure(style, "rPr")
        _set_run_font(rpr, font)
        _set_run_color_black(rpr)
        if font == "黑体":
            _ensure(rpr, "b")
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def _patch_doc_defaults(root: ET.Element) -> None:
    doc_defaults = _ensure(root, "docDefaults")
    rpr_default = _ensure(doc_defaults, "rPrDefault")
    rpr = _ensure(rpr_default, "rPr")
    _set_run_font(rpr, "宋体")
    _set_run_color_black(rpr)


def _patch_document_xml(data: bytes) -> bytes:
    root = ET.fromstring(data)
    _force_existing_colors_black(root)
    for paragraph in root.iter(_w("p")):
        _normalize_block_text_paragraph(paragraph)
        font = "黑体" if _paragraph_is_heading(paragraph) else "宋体"
        for run in paragraph.findall(_w("r")):
            _format_run(run, font, bold=font == "黑体")
    _patch_table_header_rows(root)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def _normalize_block_text_paragraph(paragraph: ET.Element) -> None:
    ppr = paragraph.find(_w("pPr"))
    if ppr is None:
        return

    pstyle = ppr.find(_w("pStyle"))
    style_value = (pstyle.attrib.get(_w("val"), "") if pstyle is not None else "").lower()
    if style_value.startswith("block"):
        ppr.remove(pstyle)
        for ind in list(ppr.findall(_w("ind"))):
            ppr.remove(ind)


def _patch_table_header_rows(root: ET.Element) -> None:
    for table in root.iter(_w("tbl")):
        _set_table_outer_borders(table)
        rows = list(table.findall(_w("tr")))
        for row_idx, row in enumerate(rows):
            if row_idx == 0 or _row_is_table_header(row):
                for run in row.iter(_w("r")):
                    _format_run(run, "黑体", bold=True)
        if rows:
            _set_row_cell_border(rows[0], "top")
            _set_row_cell_border(rows[-1], "bottom")


def _set_table_outer_borders(table: ET.Element) -> None:
    tblpr = table.find(_w("tblPr"))
    if tblpr is None:
        tblpr = ET.Element(_w("tblPr"))
        table.insert(0, tblpr)
    borders = _ensure(tblpr, "tblBorders")
    _set_border(_ensure(borders, "top"))
    _set_border(_ensure(borders, "bottom"))


def _set_row_cell_border(row: ET.Element, edge: str) -> None:
    for cell in row.findall(_w("tc")):
        tcpr = cell.find(_w("tcPr"))
        if tcpr is None:
            tcpr = ET.Element(_w("tcPr"))
            cell.insert(0, tcpr)
        borders = _ensure(tcpr, "tcBorders")
        _set_border(_ensure(borders, edge))


def _set_border(border: ET.Element) -> None:
    border.set(_w("val"), "single")
    border.set(_w("sz"), "8")
    border.set(_w("space"), "0")
    border.set(_w("color"), "000000")


def _row_is_table_header(row: ET.Element) -> bool:
    trpr = row.find(_w("trPr"))
    return trpr is not None and trpr.find(_w("tblHeader")) is not None


def _format_run(run: ET.Element, font: str, *, bold: bool = False) -> None:
    rpr = run.find(_w("rPr"))
    if rpr is None:
        rpr = ET.Element(_w("rPr"))
        run.insert(0, rpr)
    _set_run_font(rpr, font)
    _set_run_color_black(rpr)
    if bold:
        _ensure(rpr, "b")


def _paragraph_is_heading(paragraph: ET.Element) -> bool:
    ppr = paragraph.find(_w("pPr"))
    if ppr is None:
        return False
    pstyle = ppr.find(_w("pStyle"))
    if pstyle is None:
        return False
    return (pstyle.attrib.get(_w("val"), "") or "").lower().startswith("heading")


def _is_heading_style(style_id: str, style_name: str) -> bool:
    return style_id.lower().startswith("heading") or style_name.lower().startswith("heading")


def _set_run_font(rpr: ET.Element, font: str) -> None:
    fonts = rpr.find(_w("rFonts"))
    if fonts is None:
        fonts = ET.Element(_w("rFonts"))
        rpr.insert(0, fonts)
    fonts.set(_w("ascii"), EN_FONT)
    fonts.set(_w("hAnsi"), EN_FONT)
    fonts.set(_w("cs"), EN_FONT)
    fonts.set(_w("eastAsia"), font)


def _set_run_color_black(rpr: ET.Element) -> None:
    color = rpr.find(_w("color"))
    if color is None:
        color = ET.SubElement(rpr, _w("color"))
    _make_color_black(color)


def _force_existing_colors_black(root: ET.Element) -> None:
    for color in root.iter(_w("color")):
        _make_color_black(color)


def _make_color_black(color: ET.Element) -> None:
    color.set(_w("val"), "000000")
    for attr in ("themeColor", "themeTint", "themeShade", "themeFill", "themeFillTint", "themeFillShade"):
        color.attrib.pop(_w(attr), None)


def _ensure(parent: ET.Element, local_name: str) -> ET.Element:
    child = parent.find(_w(local_name))
    if child is None:
        child = ET.SubElement(parent, _w(local_name))
    return child


def _w(local_name: str) -> str:
    return f"{{{W_NS}}}{local_name}"


def _is_pipe_row(line: str) -> bool:
    cells = _pipe_cells(line)
    return line.startswith("|") and line.endswith("|") and len(cells) >= 2


def _is_blockquote_line(line: str) -> bool:
    return bool(re.match(r"^\s*>", line))


def _strip_blockquote_marker(line: str) -> str:
    return re.sub(r"^\s*(?:>\s?)+", "", line)


def _is_horizontal_rule(line: str) -> bool:
    return bool(re.fullmatch(r"\s{0,3}(?:(?:-\s*){3,}|(?:\*\s*){3,}|(?:_\s*){3,})", line))


def _is_separator_row(line: str) -> bool:
    if not _is_pipe_row(line):
        return False
    return all(re.fullmatch(r":?-{3,}:?", cell.strip() or "") for cell in _pipe_cells(line))


def _pipe_cells(line: str) -> list[str]:
    if not (line.startswith("|") and line.endswith("|")):
        return []
    return [cell.strip() for cell in line.strip().strip("|").split("|")]
