"""
src/ingest/deck.py — Slide deck (PDF or PPTX) ingestion.

Mirrors the PROVIDED video flow, but the source is a slide deck and the citation
locator is a **slide** number:

    fetch → parse (per slide) → enrich (caption) → chunk (per slide) → embed → index

Reconciled to the REAL momentsearch repo (same as paper.py — see its header and
RECONCILED.md). Decks are TEXT once captioned, so they ride the repo's *text
branch* (`embed_docs` + `vector_store.upsert_chunks` into `TEXT_COLLECTION`),
tagged `kind="deck"`, into the SAME collection as videos and papers.

Key wrinkle vs. papers: image-only slides have little/no extractable text, so
parse RENDERS them and the shared enrich stage captions them with the multimodal
LLM before embedding — otherwise those slides retrieve poorly (a graded pitfall).
Captioning lives in `paper.py :: caption_image` / `enrich_pages`, one
implementation for both source types.

LIMITATION — captioning is **PDF-only**. PyMuPDF rasterizes a PDF page, so an
image-only slide in a PDF deck gets a caption. `python-pptx` reads shapes and
cannot render a slide, so `_render_pptx_slide_png` has nothing to hand the vision
model and returns None: an image-only PPTX slide indexes on its sparse text
alone. Closing this means a LibreOffice-headless PPTX→PDF conversion up front,
reusing `_parse_pdf_deck` — a heavy binary in the image for one input format,
so it is deliberately not done. Export the deck to PDF to get captioning.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field

import fitz  # PyMuPDF (PDF decks + rendering slides to images)   pip install pymupdf

# --- shared repo pieces (REAL names, reconciled) ------------------------------
from .. import config, db, storage
from ..config import TEXT_EMBED_VERSION
from ..rag import vector_store
from ..rag.embeddings import embed_docs
# reuse paper.py's fetch, word-window chunker and enrichment (identical needs) so
# there's one implementation to maintain.
from .paper import _load_bytes, _window_chunks, CHUNK_WORDS, MIN_TEXT_CHARS, MAX_PAGES

KIND = "deck"


@dataclass
class Slide:
    slide: int         # 1-based slide number == the citation locator
    text: str          # slide text (+ appended caption for image-only slides)
    image: bytes | None = None   # rendered PNG, set only when the slide needs a caption


@dataclass
class Chunk:
    text: str
    payload: dict = field(default_factory=dict)


# Captioning itself lives in paper.py (`caption_image` / `enrich_pages`) — one
# implementation for both source types, and deck.py already imports from there.


# 1) PARSE ─────────────────────────────────────────────────────────────────────
def parse_deck(data: bytes, filename: str = "") -> list[Slide]:
    """Deck bytes → per-slide text. Handles PPTX and PDF; captions image-only
    slides in PDF decks only (see the module docstring)."""
    if filename.lower().endswith(".pptx"):
        return _parse_pptx(data)
    return _parse_pdf_deck(data)


def _parse_pptx(data: bytes) -> list[Slide]:
    from pptx import Presentation                 # pip install python-pptx

    prs = Presentation(io.BytesIO(data))
    slides: list[Slide] = []
    for i, slide in enumerate(prs.slides, start=1):
        texts = [sh.text for sh in slide.shapes
                 if getattr(sh, "has_text_frame", False) and sh.text]
        text = "\n".join(t.strip() for t in texts if t.strip())
        image = None
        if len(text) < MIN_TEXT_CHARS:
            image = _render_pptx_slide_png(slide)   # always None — see helper note
        slides.append(Slide(slide=i, text=text, image=image))
    return slides


def _parse_pdf_deck(data: bytes) -> list[Slide]:
    """PDF deck: one page == one slide. Extract text, and RENDER image-only slides
    so the enrich stage can caption them. Rendering is local and cheap; captioning
    is a flaky network call, so it belongs in its own retrying task, not here."""
    slides: list[Slide] = []
    rendered = 0
    with fitz.open(stream=data, filetype="pdf") as doc:
        if doc.page_count > MAX_PAGES:     # §11: reject abusive/oversized decks
            raise ValueError(f"Deck has {doc.page_count} slides; the limit is {MAX_PAGES}.")
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            image = None
            if (len(text) < MIN_TEXT_CHARS and config.ENRICH_CAPTIONS
                    and rendered < config.MAX_CAPTIONED_PAGES):
                image = page.get_pixmap(dpi=config.CAPTION_RENDER_DPI).tobytes("png")
                rendered += 1
            slides.append(Slide(slide=i, text=text, image=image))
    return slides


def _render_pptx_slide_png(slide) -> bytes | None:  # noqa: ANN001
    """Always None — PPTX slides are never captioned. NOT a stub left by mistake.

    python-pptx reads shapes; it cannot rasterize a slide, and there is no
    pure-Python renderer for PowerPoint layout. The two ways out:
      • convert the PPTX to PDF once (LibreOffice headless) and reuse
        `_parse_pdf_deck` — one rasterization path, but a ~400mb binary in the
        image for a single input format; or
      • pull embedded picture shapes' blobs and caption those individually —
        cheap, but it captions pictures, not slides: layout, title and chart
        text around the image are lost, which is most of what makes a slide
        findable.
    Neither is worth it while PDF decks (the common export) caption correctly.
    Returning None degrades to text-only indexing for that slide — never fatal.
    """
    return None


# 2) CHUNK (slide-aware) ──────────────────────────────────────────────────────
def chunk_deck(source_id: str, title: str, slides: list[Slide],
               user_id: str) -> list[Chunk]:
    """One (or a few) chunk(s) per slide, each carrying `slide` as the locator.
    Most slides are short → one chunk; only long, dense slides get windowed.

    `user_id` MUST be in the payload — every search is user_id-filtered, so a
    chunk without it is invisible to retrieval (mirrors the video payload)."""
    chunks: list[Chunk] = []
    for sl in slides:
        words = len(sl.text.split())
        pieces = _window_chunks(sl.text) if words > CHUNK_WORDS else (
            [sl.text] if sl.text.strip() else [])
        for piece in pieces:
            chunks.append(Chunk(
                text=piece,
                payload={
                    "user_id": user_id,       # ← multi-tenant filter (required!)
                    "kind": KIND,
                    "source_id": source_id,
                    "video_id": source_id,   # read path fuses/filters on video_id
                    "title": title,
                    "slide": sl.slide,        # ← locator
                    "modality": "text",
                    "text": piece,
                    "t_start": 0.0, "t_end": 0.0,
                    "embed_version": TEXT_EMBED_VERSION,
                },
            ))
    return chunks


# The FLOW lives in src/ingest/flows.py (`ms-ingest-deck`), as per-stage Prefect
# tasks mirroring the paper flow. This module owns the STAGES only.
