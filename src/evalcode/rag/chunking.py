"""Deterministic chunking helpers.

These functions turn raw text into bounded pieces that a MiniLM-class embedder
can handle (its ~256 word-piece window means we keep chunks around 1000 chars,
with the most informative content — the signature — first).
"""

from __future__ import annotations

import hashlib
import re

__all__ = ["stable_id", "truncate_text", "split_markdown"]

_ELLIPSIS = "…"


def stable_id(*parts: str) -> str:
    """Return a stable 16-char hex id derived from the given parts.

    A separator is inserted between parts so that ``("ab", "c")`` and
    ``("a", "bc")`` do not collide.
    """
    joined = "\x00".join(parts)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:16]


def truncate_text(text: str, max_chars: int) -> str:
    """Cut ``text`` to at most ``max_chars``, preferring a clean boundary.

    Preference order: sentence end, newline, then word boundary. When the text
    is actually cut, a single ``…`` (U+2026) is appended; the result is always
    no longer than ``max_chars``. Text already within the limit is returned
    unchanged.
    """
    if max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text

    window = text[: max_chars - 1]  # reserve one slot for the ellipsis
    cut = _pick_truncation(window)
    cut = max(cut, 1)
    return window[:cut].rstrip() + _ELLIPSIS


def _pick_truncation(window: str) -> int:
    """Choose a truncation index inside ``window``."""
    last_sentence: re.Match[str] | None = None
    for match in re.finditer(r"[.!?](?:\s|$)", window):
        last_sentence = match
    if last_sentence is not None:
        return last_sentence.end()

    nl = window.rfind("\n")
    if nl > 0:
        return nl + 1
    space = window.rfind(" ")
    if space > 0:
        return space + 1
    return len(window)


# --- Markdown splitting ----------------------------------------------------

_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s")


def split_markdown(
    text: str,
    max_chars: int = 800,
    overlap: int = 100,
) -> list[str]:
    """Split markdown/text into bounded pieces, keeping heading context.

    Strategy: first split on headings (each piece keeps the current heading at
    its top), then pack content up to ``max_chars`` preferring paragraph,
    line, and word boundaries. Consecutive pieces of an over-long section share
    an ``overlap``-character tail so context is not lost at the boundary. No
    piece is empty; a single line longer than ``max_chars`` is returned whole
    rather than broken mid-line. CRLF line endings are treated as LF.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    out: list[str] = []
    for heading, body in _split_into_sections(text):
        out.extend(_split_section(heading, body, max_chars, overlap))
    return [piece for piece in out if piece.strip()]


def _split_into_sections(text: str) -> list[tuple[str, str]]:
    """Split into ``(heading, body)`` sections on heading lines."""
    sections: list[tuple[str, str]] = []
    heading = ""
    buffer: list[str] = []
    for line in text.split("\n"):
        if _HEADING_RE.match(line):
            if heading or "".join(buffer).strip():
                sections.append((heading, "\n".join(buffer)))
            heading = line.strip()
            buffer = []
        else:
            buffer.append(line)
    if heading or "".join(buffer).strip():
        sections.append((heading, "\n".join(buffer)))
    return [(h, b) for h, b in sections if h.strip() or b.strip()]


def _split_section(heading: str, body: str, max_chars: int, overlap: int) -> list[str]:
    """Split one section, re-attaching the heading to every piece."""
    body = body.strip()
    if not body:
        return [heading] if heading and len(heading) <= max_chars else []

    prefix = heading + "\n\n" if heading else ""
    # Fast path: the whole section fits.
    if len(prefix) + len(body) <= max_chars:
        return [prefix + body]

    budget = max(max_chars - len(prefix), 1)
    pieces = _split_into_pieces(body, budget, overlap)
    return [prefix + piece for piece in pieces]


def _split_into_pieces(body: str, max_chars: int, overlap: int) -> list[str]:
    """Slice ``body`` into pieces of at most ``max_chars`` with an overlap."""
    n = len(body)
    pieces: list[str] = []
    pos = 0
    while pos < n:
        end = pos + max_chars
        if end >= n:
            pieces.append(body[pos:].strip())
            break

        cut = _find_boundary(body, pos + 1, end)
        extended = False
        if cut == end and _is_unsplittable(body, pos, end):
            # A single line/word runs past the window: keep it whole even if
            # it exceeds the limit (the documented exception).
            nl = body.find("\n", end)
            cut = nl + 1 if nl != -1 else n
            extended = True
        if cut <= pos:
            cut = end

        pieces.append(body[pos:cut].strip())
        if cut >= n:
            break
        # Overlap into the next piece, unless we just emitted a long line.
        next_pos = cut if extended else cut - overlap
        pos = next_pos if next_pos > pos else cut
    return [piece for piece in pieces if piece]


def _find_boundary(text: str, lo: int, hi: int) -> int:
    """Find the last clean boundary in ``text[lo:hi]`` (else ``hi``)."""
    segment = text[lo:hi]
    for marker, step in (("\n\n", 2), ("\n", 1), (" ", 1)):
        idx = segment.rfind(marker)
        if idx > 0:
            return lo + idx + step
    return hi


def _is_unsplittable(text: str, lo: int, hi: int) -> bool:
    """True when ``text[lo:hi]`` is one unbroken token with no line break."""
    segment = text[lo:hi]
    return "\n" not in segment and " " not in segment
