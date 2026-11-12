"""Deployment answer parser (Python mirror of cpp/tools/student_v1.cc parse_answer/fill_box).

Rules (gpi-3s deploy-parse-v1), identical to the C++ implementation:

  box_strict         C C C C B E     deployment valid, original protocol valid
  box_compact        C C C C E       deployment valid, original protocol INVALID (no B)
  none_strict        N B E           deployment valid, none, protocol valid
  none_compact       N E             deployment valid, none, protocol INVALID
  unexpected_token   four coords followed by anything other than B E / E, or junk after N
  incomplete         no <im_end> (truncated at the token cap), or fewer than 4 tokens
  bad_prefix         four-token prefix is not four coordinates
  trailing_after_end tokens after <|im_end|> (duplicate/stray terminator)
  degenerate_box     coordinates map to non-positive area / non-finite
  empty              no tokens

historical_extractable = old rule: first four tokens are coords and the last token is
<|im_end|> (independent of what sits in between). Reported for comparison only.
"""

COORD_BASE, COORD_LAST = 151670, 152669
BOX_END, IM_END, NONE = 151649, 151645, 151669
CANVAS = 384


def is_coord(t):
    return COORD_BASE <= t <= COORD_LAST


def parse_answer(tokens):
    out = dict(status="empty", valid=False, strict=False, none=False, has_box_end=False,
               ended=False, historical_extractable=False, failure="",
               box=[0.0, 0.0, 0.0, 0.0], box_ok=False, out_of_image=False, grid=None)
    if not tokens:
        out["failure"] = "no_tokens"
        return out

    out["has_box_end"] = BOX_END in tokens
    out["ended"] = tokens[-1] == IM_END

    if IM_END in tokens:
        first_end = tokens.index(IM_END)
        if not out["ended"] or first_end != len(tokens) - 1:
            out["status"] = "trailing_after_end"
            out["failure"] = "tokens_after_im_end"
            return out

    n = len(tokens) - 1 if out["ended"] else len(tokens)
    four = n >= 4 and all(is_coord(t) for t in tokens[:4])
    out["historical_extractable"] = four and out["ended"]

    if n >= 1 and tokens[0] == NONE:
        if not out["ended"]:
            out["status"] = "incomplete"
            out["failure"] = "no_im_end"
            return out
        if n == 1:
            out["status"] = "none_compact"
            out["valid"] = True
            out["none"] = True
            return out
        if n == 2 and tokens[1] == BOX_END:
            out["status"] = "none_strict"
            out["valid"] = True
            out["strict"] = True
            out["none"] = True
            return out
        out["status"] = "unexpected_token"
        out["failure"] = "tokens_after_none"
        return out

    if not four:
        if n < 4:
            out["status"] = "bad_prefix" if out["ended"] else "incomplete"
            out["failure"] = "fewer_than_4_tokens"
        else:
            out["status"] = "bad_prefix"
            out["failure"] = "first_four_not_coords"
        return out

    if not out["ended"]:
        out["status"] = "incomplete"
        out["failure"] = "no_im_end"
        return out
    if n == 4:
        out["status"] = "box_compact"
        out["valid"] = True
        return out
    if n == 5 and tokens[4] == BOX_END:
        out["status"] = "box_strict"
        out["valid"] = True
        out["strict"] = True
        return out
    out["status"] = "unexpected_token"
    out["failure"] = "tokens_between_coords_and_end"
    return out


def letterbox(w, h):
    scale = min(CANVAS / w, CANVAS / h)
    rw = max(1, int(w * scale + 0.5))
    rh = max(1, int(h * scale + 0.5))
    return dict(rw=rw, rh=rh, left=(CANVAS - rw) // 2, top=(CANVAS - rh) // 2,
                sx=rw / w, sy=rh / h)


def fill_box(p, tokens, w, h, lb=None):
    """Map the four coordinate tokens to original-image pixels (no clipping)."""
    if not p["valid"] or p["none"]:
        return p
    lb = lb or letterbox(w, h)
    v = []
    for i, t in enumerate(tokens[:4]):
        c = (t - COORD_BASE) / 999.0 * CANVAS
        v.append((c - (lb["left"] if i % 2 == 0 else lb["top"])) /
                 (lb["sx"] if i % 2 == 0 else lb["sy"]))
    p["grid"] = [t - COORD_BASE for t in tokens[:4]]
    if any(x != x or abs(x) == float("inf") for x in v):
        p["valid"] = False
        p["status"] = "degenerate_box"
        p["failure"] = "non_finite"
        return p
    p["box"] = v
    if v[2] <= v[0] or v[3] <= v[1]:
        p["valid"] = False
        p["status"] = "degenerate_box"
        p["failure"] = "non_positive_area"
        return p
    p["box_ok"] = True
    p["out_of_image"] = bool(v[0] < 0 or v[1] < 0 or v[2] > w or v[3] > h)
    return p


def parse_and_map(tokens, w, h, lb=None):
    return fill_box(parse_answer(tokens), tokens, w, h, lb)


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + bb - inter)
