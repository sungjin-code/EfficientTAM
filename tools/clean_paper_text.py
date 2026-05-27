"""Clean OCR/PDF line breaks in paper.txt while preserving paper details.

The script keeps the agent guide at the top of paper.txt, then dewraps the
paper body under the "Original Paper Text" marker. It removes page numbers and
CVF watermark lines, joins hyphenated OCR line breaks, and leaves table/equation
blocks mostly line-preserved so numerical details are not mangled.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


MARKER = "Original Paper Text\n==================="
WATERMARK_PREFIXES = (
    "This ICCV paper is the Open Access version",
    "Except for this watermark",
    "the final published version",
)
SECTION_RE = re.compile(r"^\d+(?:\.\d+)?\. .+")
PAGE_RE = re.compile(r"^\d{5}$")
METHOD_ROW_RE = re.compile(
    r"^(STCN|RDE|XMem|DEVA|DEV A|Cutie-base\+?|SAM 2|SAM\+|EfficientTAM|HQ-SAM|SAM \(ViT)"
)


def _normalize_inline(text: str) -> str:
    replacements = {
        "[ ": "[",
        " ]": "]",
        "DA VIS": "DAVIS",
        "DEV A": "DEVA",
        "mean-absolution-error": "mean-absolute-error",
        "illusrates": "illustrates",
        "and and": "and",
        "J &F": "J&F",
        "J & F": "J&F",
        "w.r.t original": "w.r.t. original",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"\[\s+", "[", text)
    text = re.sub(r"\s+\]", "]", text)
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    text = re.sub(r"([A-Za-z])\s+([,.;:])", r"\1\2", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def _is_skip_line(line: str) -> bool:
    stripped = line.strip()
    return PAGE_RE.fullmatch(stripped) is not None or stripped.startswith(WATERMARK_PREFIXES)


def _is_heading(line: str) -> bool:
    stripped = line.strip()
    if stripped in {"Abstract", "References", "Acknowledgements"}:
        return True
    return SECTION_RE.match(stripped) is not None


def _merge_split_headings(lines: list[str]) -> list[str]:
    merged: list[str] = []
    i = 0
    while i < len(lines):
        current = lines[i].strip()
        next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
        if (
            SECTION_RE.match(current)
            and current.endswith((" and", " of", " for"))
            and next_line
            and len(next_line) < 60
            and not next_line.endswith(".")
        ):
            merged.append(current + " " + next_line)
            i += 2
            continue
        merged.append(lines[i])
        i += 1
    return merged


def _is_short_table_header(line: str) -> bool:
    stripped = line.strip()
    if len(stripped) > 80 or "." in stripped:
        return False
    headers = {
        "Method J&F G",
        "Method 1-click 3-click 5-click bounding box ground-truth mask",
        "Parameters",
        "(M)",
        "FPS Latency (ms)",
        "MOSE",
        "val",
        "LVOS",
        "SA-V",
        "test",
        "YTVOS",
        "2019 val A100 iPhone15",
        "A100 iPhone15",
        "Number of annotated frames with 3-click",
        "Model SA-23 All SA-23 Image SA-23 Video 14 new Video",
    }
    if stripped in headers:
        return True
    if re.fullmatch(r"(?:\d+\s+){2,}\d+", stripped):
        return True
    return False


def _is_table_row(line: str) -> bool:
    stripped = line.strip()
    if not METHOD_ROW_RE.match(stripped):
        return False
    return len(re.findall(r"\d+(?:\.\d+)?", stripped)) >= 3


def _is_equation_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if len(stripped) > 90:
        return False
    starts = (
        "softmax",
        "where ",
        "Assume ",
        "Then ",
        "With Eq.",
        "There is a constant",
        "avoiding ",
        "performance.",
        "˜",
        "¯",
        "Q ",
        "K ",
        "V ",
        "A =",
        "˜V =",
        "˜K =",
        "d",
        "p=",
        "q=",
        "i =",
        "j =",
    )
    math_symbols = sum(stripped.count(ch) for ch in "˜¯√≤≥∈{}=×")
    if stripped.startswith(starts) and math_symbols:
        return True
    if math_symbols >= 2 and len(stripped.split()) <= 10:
        return True
    if stripped in {",", "]", ")", "", "!", "d"}:
        return True
    return False


def _hyphen_should_stay(text_before_hyphen: str, next_line: str) -> bool:
    match = re.search(r"([A-Za-z]+)-$", text_before_hyphen)
    if not match:
        return False
    prev = match.group(1).lower()
    next_word_match = re.match(r"([A-Za-z]+)", next_line)
    next_word = next_word_match.group(1).lower() if next_word_match else ""
    keep_prefixes = {
        "semi",
        "zero",
        "end",
        "real",
        "state",
        "non",
        "multi",
        "pre",
        "near",
        "on",
        "high",
        "low",
        "first",
        "single",
        "open",
        "object",
        "cross",
        "long",
        "one",
        "few",
    }
    known_terms = {
        ("state", "of"),
        ("end", "to"),
        ("cross", "attention"),
        ("object", "level"),
        ("first", "frame"),
        ("single", "scale"),
        ("high", "resolution"),
        ("low", "light"),
        ("on", "device"),
        ("near", "real"),
        ("pre", "trained"),
        ("multi", "stage"),
        ("semi", "supervised"),
        ("zero", "shot"),
    }
    return prev in keep_prefixes or (prev, next_word) in known_terms


def _render_paragraph(lines: list[str]) -> str:
    text = ""
    for raw in lines:
        line = raw.strip()
        if not text:
            text = line
            continue
        if text.endswith("-"):
            if _hyphen_should_stay(text, line):
                text += line
            else:
                text = text[:-1] + line
        else:
            text += " " + line
    return _normalize_inline(text)


def _fix_remaining_hyphen_newlines(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        prev = match.group(1)
        next_word = match.group(2)
        if _hyphen_should_stay(prev + "-", next_word):
            return prev + "-" + next_word
        return prev + next_word

    return re.sub(r"([A-Za-z]{3,})-\n([a-z][A-Za-z]*)", replace, text)


def _flush_paragraph(out: list[str], paragraph: list[str]) -> None:
    if not paragraph:
        return
    rendered = _render_paragraph(paragraph)
    if rendered:
        out.append(rendered)
    paragraph.clear()


def _append_blank(out: list[str]) -> None:
    if out and out[-1] != "":
        out.append("")


def clean_body(text: str) -> str:
    lines = _merge_split_headings(text.splitlines())
    out: list[str] = []
    paragraph: list[str] = []
    metadata = True
    skipped_layout_line = False

    for raw_line in lines:
        line = raw_line.strip()
        if _is_skip_line(line):
            skipped_layout_line = True
            continue
        if not line:
            if skipped_layout_line:
                skipped_layout_line = False
                continue
            _flush_paragraph(out, paragraph)
            _append_blank(out)
            continue
        skipped_layout_line = False

        normalized_line = _normalize_inline(line)

        if metadata:
            out.append(normalized_line)
            if normalized_line == "Abstract":
                metadata = False
                _append_blank(out)
            continue

        if _is_heading(normalized_line):
            _flush_paragraph(out, paragraph)
            _append_blank(out)
            out.append(normalized_line)
            _append_blank(out)
            continue

        if normalized_line.startswith(("Figure ", "Table ")):
            _flush_paragraph(out, paragraph)
            paragraph.append(normalized_line)
            continue

        if _is_short_table_header(normalized_line) or _is_table_row(normalized_line):
            _flush_paragraph(out, paragraph)
            out.append(normalized_line)
            continue

        if _is_equation_line(normalized_line):
            _flush_paragraph(out, paragraph)
            out.append(normalized_line)
            continue

        if normalized_line.startswith("• "):
            _flush_paragraph(out, paragraph)
            paragraph.append(normalized_line)
            continue

        paragraph.append(normalized_line)

    _flush_paragraph(out, paragraph)
    while out and out[-1] == "":
        out.pop()
    return _fix_remaining_hyphen_newlines("\n".join(out)) + "\n"


def clean_paper_text(text: str) -> str:
    if MARKER not in text:
        return clean_body(text)
    prefix, body = text.split(MARKER, 1)
    cleaned_body = clean_body(body.lstrip("\n"))
    return prefix.rstrip() + "\n\n" + MARKER + "\n\n" + cleaned_body


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean OCR line breaks in paper.txt")
    parser.add_argument("path", nargs="?", default="paper.txt")
    parser.add_argument("--output", default=None)
    parser.add_argument("--in-place", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.path)
    cleaned = clean_paper_text(path.read_text())
    if args.in_place:
        path.write_text(cleaned)
    elif args.output:
        Path(args.output).write_text(cleaned)
    else:
        print(cleaned, end="")


if __name__ == "__main__":
    main()
