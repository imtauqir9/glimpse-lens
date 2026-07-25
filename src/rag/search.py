"""Read path: question -> retrieve -> gate -> cited answer (or honest abstain).

Retrieval is milliseconds; the multimodal LLM call is seconds and dominates
cost. So the shape is a confidence funnel: fetch KNN_K candidates, collapse
temporal near-duplicates, trim to TOP_K, and — Gate 1 — if even the best
score is below CONFIDENCE_THRESHOLD, abstain WITHOUT calling the LLM. That
one free check kills most hallucination risk. Generated answers get their
[n] citations validated; invented references are stripped.

# +DOC — This file is the upstream momentsearch read path, extended for
# multi-source citations (paper page / deck slide) alongside video moments.
# Every change is marked `# +DOC`; video behavior is byte-for-byte unchanged.
# Papers/decks arrive via the SAME text branch as transcripts (they carry
# `kind`, `page`/`slide` in their payload and t_start=0.0). Because a document's
# chunks all share t=0, time-window fusion would collapse a whole paper into one
# citation — so doc hits bucket by their LOCATOR (page/slide) instead of by time.
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .. import config, db, llm, storage
from ..config import (BRANCH_TOP_K, CONFIDENCE_THRESHOLD, CROSS_MODAL_BOOST,
                      FUSION_WINDOW_S, RRF_K, TEXT_CONFIDENCE_THRESHOLD, TOP_K)
from . import vector_store
from .embeddings import embed_query, embed_text

ABSTAIN = ("I couldn't find that in your sources — nothing indexed looks "  # +DOC: "videos"->"sources"
           "related to the question (neither what's on screen nor what's said).")


def _seconds(ms: int) -> str:
    s = ms // 1000
    return f"{s // 60:02d}:{s % 60:02d}"


# +DOC ─ locator identity of a hit. None == a video (time-based fusion);
#        ("paper", page) / ("deck", slide) == a document (locator-based fusion).
def _loc_key(h: dict) -> tuple[str, Any] | None:
    kind = h.get("kind", "video")
    if kind == "paper":
        return ("paper", h.get("page"))
    if kind == "deck":
        return ("deck", h.get("slide"))
    return None


def _fuse(visual_hits: list[dict], text_hits: list[dict]) -> list[dict]:
    """Reciprocal-Rank-Fusion of the two branches into time windows.

    Raw scores are incomparable (CLIP ~0.3 vs bge ~0.7), so we rank each branch
    on its own and score by rank: rrf = 1/(RRF_K + rank). Then we bucket hits
    within FUSION_WINDOW_S seconds of each other (same video) into one 'moment',
    sum their rrf, and boost windows where BOTH modalities agree — two
    independent signals pointing at the same instant is the strongest evidence.

    # +DOC — Document hits (kind=paper|deck) bucket by their locator instead of
    # by time: two different pages of the same paper are two windows, never one.
    """
    def ranked(hits, modality):
        out = []
        for rank, h in enumerate(hits):
            t = float(h.get("t_start", h.get("ms", 0) / 1000.0))
            out.append({**h, "modality": modality, "rrf": 1.0 / (RRF_K + rank), "t": t})
        return out

    windows: list[dict] = []
    # Hits arrive best-first (rrf desc), so the first hit landing in a window for
    # a given modality is that modality's best hit there.
    for h in sorted(ranked(visual_hits, "frame") + ranked(text_hits, "text"),
                    key=lambda x: x["rrf"], reverse=True):
        lk = _loc_key(h)  # +DOC
        if lk is None:    # +DOC: video — original time-proximity bucketing
            w = next((w for w in windows if w.get("loc") is None
                      and w["video_id"] == h["video_id"]
                      and abs(w["t"] - h["t"]) <= FUSION_WINDOW_S), None)
        else:             # +DOC: document — bucket by (video_id, locator)
            w = next((w for w in windows if w.get("loc") == lk
                      and w["video_id"] == h["video_id"]), None)
        if w is None:
            w = {"video_id": h["video_id"], "t": h["t"], "rrf": 0.0,
                 "modalities": set(), "frame": None, "text": None,
                 "loc": lk,                                   # +DOC
                 "kind": h.get("kind", "video"),              # +DOC
                 "page": h.get("page"), "slide": h.get("slide")}  # +DOC
            windows.append(w)
        w["modalities"].add(h["modality"])
        slot = "frame" if h["modality"] == "frame" else "text"
        # Keep only the BEST hit per modality. Summing every hit would let a
        # burst of near-identical frames clustered in one 15s window inflate its
        # score past a genuine frame+transcript match — the bug that ranked a
        # silent frame-burst above the moment that actually answered.
        if w[slot] is None:
            w[slot] = h
    for w in windows:
        # Score = best frame + best transcript hit; ×boost when BOTH modalities
        # agree at this instant (two independent signals = strongest evidence).
        # +DOC: unreachable for a document — it has no visual branch — so this
        # ranks video moments against each OTHER honestly but handicaps papers
        # and decks against them. _diversify() compensates on the way out.
        w["rrf"] = (w["frame"]["rrf"] if w["frame"] else 0.0) + \
                   (w["text"]["rrf"] if w["text"] else 0.0)
        if {"frame", "text"} <= w["modalities"]:
            w["rrf"] *= CROSS_MODAL_BOOST
    windows.sort(key=lambda w: w["rrf"], reverse=True)
    return windows


# +DOC ─ cross-source coverage in the returned top-k.
def _kind_of(w: dict) -> str:
    return w.get("kind") or "video"


def _credible(w: dict) -> bool:
    """Does this window stand on its own evidence?

    Deliberately the SAME test as Gate 1 in ask(): each branch judged against its
    own raw-score threshold, because CLIP cosines (~0.2-0.35) and bge cosines
    (~0.5-0.7) are not on one scale. If a window wouldn't survive the abstain
    gate, it has no business being promoted into the answer for variety's sake.
    """
    fr = float((w.get("frame") or {}).get("score") or 0.0)
    tx = float((w.get("text") or {}).get("score") or 0.0)
    return fr >= CONFIDENCE_THRESHOLD or tx >= TEXT_CONFIDENCE_THRESHOLD


def _diversify(windows: list[dict], k: int) -> list[dict]:
    """Make sure a kind with real evidence isn't ranked out of the top-k.

    Why this is needed at all: CROSS_MODAL_BOOST multiplies a window scored by
    BOTH branches, and a paper or deck lives only in the text branch — it cannot
    reach {frame, text} no matter how relevant it is. So the boost, meant to
    separate corroborated video moments from uncorroborated ones, also silently
    demotes every document beneath every corroborated video moment. The boost
    still does its intra-video job; this stops that side effect from costing a
    document its citation in a mixed corpus (design §6's definition of done: one
    query returns a video moment, a paper passage AND a slide).

    Promotion is bounded and earned — at most CROSS_SOURCE_RESERVED slots, only
    for kinds missing from the top-k, only for windows that pass _credible, and
    never at the cost of the #1 result. An irrelevant kind is left out.
    """
    top = windows[:k]
    if not config.CROSS_SOURCE_DIVERSITY or len(windows) <= k:
        return top
    budget = min(config.CROSS_SOURCE_RESERVED, max(0, k - 1))
    if budget <= 0:
        return top

    present = {_kind_of(w) for w in top}
    promoted: list[dict] = []
    for w in windows[k:]:
        if len(promoted) >= budget:
            break
        kind = _kind_of(w)
        if kind in present or not _credible(w):
            continue
        promoted.append(w)
        present.add(kind)          # one slot per kind, not one per hit
    if not promoted:
        return top

    keep = top[:len(top) - len(promoted)]      # drop the weakest, keep the best
    return sorted(keep + promoted, key=lambda w: w["rrf"], reverse=True)


def _deeplink(video: dict | None, video_id: str, ms: int) -> str:
    secs = ms // 1000
    if video and video.get("source") == "youtube" and video.get("url"):
        sep = "&" if "?" in video["url"] else "?"
        return f"{video['url']}{sep}t={secs}"
    return f"/api/video/{video_id}#t={secs}"


# +DOC ─ deep-link for a document citation (open the exact page/slide).
def _doc_deeplink(video: dict | None, source_id: str, kind: str,
                  page: int | None, slide: int | None) -> str:
    frag = f"page={page}" if kind == "paper" else f"slide={slide}"
    # Prefer the original document URL when we have one (arXiv etc.); PDF viewers
    # honor #page=. Otherwise a repo-served route (build the endpoint later).
    if video and video.get("url"):
        sep = "&" if "?" in video["url"] else "?"
        return f"{video['url']}{sep}{frag}" if kind == "paper" else f"{video['url']}#{frag}"
    return f"/api/document/{source_id}#{frag}"


def _thumb_url(user_id: str, video_id: str, idx: int) -> str:
    """Browser-facing thumbnail URL. Presigned GET straight to the bucket when
    the provider supports it (an <img> tag can't send auth headers); the API
    serves the bytes itself only in local-dev mode."""
    if storage.presign_capable():
        return storage.presign_get(storage.frame_key(user_id, video_id, idx))
    return f"/api/frame/{video_id}/{idx:06d}.jpg?u={user_id}"


def _media_url(video: dict | None, user_id: str, video_id: str) -> str | None:
    """Playback URL for uploaded videos (YouTube plays via its own URL)."""
    if not video or video.get("source") != "upload" or not video.get("storage_key"):
        return None
    if storage.presign_capable():
        return storage.presign_get(video["storage_key"])
    return f"/api/video/{video_id}?u={user_id}"


def retrieve(question: str, user_id: str, *, top_k: int | None = None,
             video_id: str | None = None,
             video_ids: list[str] | None = None) -> dict[str, Any]:
    """Multimodal retrieve: query BOTH branches (CLIP frames + transcript text),
    fuse by RRF into time windows, and return numbered moment-citations.

    Returns {citations, best_visual, best_text} — the two raw bests feed the
    confidence gate (RRF scores are too small to threshold on). video_ids scopes
    the search to chosen videos (UI select/unselect).

    # +DOC — Each citation now carries `kind` and a typed `locator`
    # ({start_ms} | {page} | {slide}) so the UI can render + deep-link per kind.
    """
    k = top_k or TOP_K

    # Visual branch — CLIP text→image.
    vhits = vector_store.search(embed_text(question), user_id, top_k=BRANCH_TOP_K,
                                video_id=video_id, video_ids=video_ids)
    best_visual = vhits[0]["score"] if vhits else 0.0

    # Text branch — bge query→transcript-chunk (only if transcript is enabled).
    # +DOC: papers + decks live in this SAME text collection, so they come back here.
    thits: list[dict] = []
    best_text = 0.0
    if config.ENABLE_TRANSCRIPT:
        thits = vector_store.search_text(embed_query(question), user_id,
                                         top_k=BRANCH_TOP_K, video_id=video_id,
                                         video_ids=video_ids)
        best_text = thits[0]["score"] if thits else 0.0

    # +DOC: fuse ranks by score; _diversify then guarantees a kind with real
    # evidence isn't ranked out by a boost it structurally cannot earn.
    windows = _diversify(_fuse(vhits, thits), k)
    videos = db.videos_by_ids(sorted({w["video_id"] for w in windows}))
    citations = []
    for i, w in enumerate(windows, 1):
        vid = w["video_id"]
        meta = videos.get(vid)
        fr, tx = w["frame"], w["text"]
        kind = w.get("kind") or (meta or {}).get("kind") or "video"  # +DOC

        if kind in ("paper", "deck"):  # +DOC ─ document citation (page/slide locator)
            page, slide = w.get("page"), w.get("slide")
            locator = {"page": page} if kind == "paper" else {"slide": slide}
            citations.append({
                "n": i,
                "video_id": vid,          # (== source_id; kept for shape parity)
                "source_id": vid,
                "kind": kind,
                "title": (meta or {}).get("title") or vid,
                "url": (meta or {}).get("url"),
                "source": (meta or {}).get("source") or kind,
                "locator": locator,                              # typed locator
                "locator_label": f"p.{page}" if kind == "paper" else f"slide {slide}",
                "ms": None, "timestamp": None,                   # no time for docs
                "idx": None, "thumbnail": None, "media_url": None,
                "deeplink": _doc_deeplink(meta, vid, kind, page, slide),
                "score": round(w["rrf"], 4),
                "transcript": (tx or {}).get("text"),            # the passage text
                "modalities": sorted(w["modalities"]),
            })
            continue

        # ── video moment (unchanged) ──
        # Anchor on the frame's exact timestamp when there is one (precise visual
        # seek); otherwise the transcript chunk's start.
        ms = int(fr["ms"]) if fr else int(w["t"] * 1000)
        idx = int(fr["idx"]) if fr else None
        citations.append({
            "n": i,
            "video_id": vid,
            "kind": "video",                                     # +DOC: explicit kind
            "title": (meta or {}).get("title") or vid,
            "url": (meta or {}).get("url"),
            "source": (meta or {}).get("source"),
            "locator": {"start_ms": ms},                         # +DOC: typed locator
            "ms": ms,
            "timestamp": _seconds(ms),
            "idx": idx,
            "thumbnail": _thumb_url(user_id, vid, idx) if idx is not None else None,
            "media_url": _media_url(meta, user_id, vid),
            "deeplink": _deeplink(meta, vid, ms),
            "score": round(w["rrf"], 4),
            "transcript": (tx or {}).get("text"),
            "modalities": sorted(w["modalities"]),
        })
    return {"citations": citations, "best_visual": best_visual, "best_text": best_text}


def _fallback_answer(citations: list[dict[str, Any]]) -> str:
    """No-LLM summary: rank the closest moments. Honest about being
    similarity, not synthesis."""
    top = citations[0]
    # +DOC: a document has no timestamp — describe it by its locator label.
    where_at = top.get("timestamp") or top.get("locator_label") or ""
    where = f"{top['title']} at {where_at}" if top.get("title") else where_at
    others = ", ".join(f"{(c.get('timestamp') or c.get('locator_label'))} [{c['n']}]"
                       for c in citations[1:4])
    msg = f"Closest match: {where} [{top['n']}] (similarity {top['score']})."  # +DOC: "visual "->""
    if others:
        msg += f" Other relevant results: {others}."
    return msg


_CITE_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def _validate_citations(answer: str, n_frames: int) -> str:
    """Strip invented [n] references the model has no frame for."""
    def fix(m: re.Match) -> str:
        nums = [int(x) for x in re.split(r"\s*,\s*", m.group(1))]
        valid = [str(x) for x in nums if 1 <= x <= n_frames]
        return f"[{', '.join(valid)}]" if valid else ""
    return _CITE_RE.sub(fix, answer)


def _build_moments(user_id: str, citations: list[dict[str, Any]]) -> list[dict]:
    """Turn citations into what the LLM sees: each moment carries its frame
    image (if any) and/or its transcript excerpt (if any), numbered to match.

    # +DOC — document citations have no frame; they pass their passage text as
    # the 'transcript' and their page/slide label as the 'timestamp' so the LLM
    # can reference "[n] p.7" naturally.
    """
    def frame_bytes(c):
        if c.get("idx") is None:
            return None
        try:
            return storage.get_bytes(storage.frame_key(user_id, c["video_id"], c["idx"]))
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=6) as ex:
        images = list(ex.map(frame_bytes, citations))
    return [{"image": img, "transcript": c.get("transcript"),
             "timestamp": c.get("timestamp") or c.get("locator_label") or ""}  # +DOC
            for img, c in zip(images, citations)]


def resolve_llm(user_id: str) -> tuple[llm.LLMConfig | None, str]:
    """Which model answers for this tenant: their own hosted endpoint
    (ms_user_llms — e.g. a vLLM server) first, the server-wide LLM_* env
    config as fallback. Returns (config, source) with source in
    {"user", "server", "none"}."""
    row = db.get_user_llm(user_id)
    if row and row.get("model"):
        return llm.from_row(row), "user"
    cfg = llm.env_config()
    return (cfg, "server") if cfg else (None, "none")


def ask(question: str, user_id: str, *, top_k: int | None = None,
        video_id: str | None = None,
        video_ids: list[str] | None = None) -> dict[str, Any]:
    r = retrieve(question, user_id, top_k=top_k, video_id=video_id, video_ids=video_ids)
    citations = r["citations"]
    result: dict[str, Any] = {"question": question, "citations": citations}

    if not citations:
        result.update(answer="No relevant results were found. Try ingesting a source first.",
                      llm_used=False, abstained=True)
        return result

    # Gate 1 — confidence on the RAW per-branch bests (not the RRF score).
    # Abstain only if NEITHER what's on screen nor what's said/written looks relevant.
    visual_ok = r["best_visual"] >= CONFIDENCE_THRESHOLD
    text_ok = r["best_text"] >= TEXT_CONFIDENCE_THRESHOLD
    if CONFIDENCE_THRESHOLD and not visual_ok and not text_ok:
        result.update(answer=ABSTAIN, llm_used=False, abstained=True)
        return result

    cfg, source = resolve_llm(user_id)
    if cfg is None:
        # No generative model — summarize the best matches instead of inventing.
        result.update(answer=_fallback_answer(citations), llm_used=False,
                      note=("Retrieval-only results. Connect your own model "
                            "(vLLM/Ollama/API) in settings, or set LLM_API_KEY "
                            "on the server, for a synthesized, grounded answer."))
        return result

    moments = _build_moments(user_id, citations)
    result["answer"] = _validate_citations(llm.answer(question, moments, cfg),
                                           len(citations))
    from .. import guardrails                 # +GUARD: cost visibility (§18)
    guardrails.meter_llm_call(user_id)
    result["llm_used"] = True
    result["llm_source"] = source          # "user" = their own hosted model
    result["llm_model"] = cfg.model
    return result
