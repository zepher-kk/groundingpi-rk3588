# Output protocol and parser (`deploy-parse-v1`)

Coordinates are 1000 buckets over the 384 canvas: `q = round(x / 384 * 999)`, token id
`151670 + q`. `<|none_box|>` is 151669, `<|box_end|>` 151649, `<|im_end|>` 151645.

## Status table

Let `C` = coordinate token, `B` = `<|box_end|>`, `E` = `<|im_end|>`, `N` = `<|none_box|>`.

| Generated sequence | status | deployment valid | original strict |
| --- | --- | :---: | :---: |
| `C C C C E` | `box_compact` | yes | no (missing `B`) |
| `C C C C B E` | `box_strict` | yes | yes |
| `N E` | `none_compact` | yes (none) | no |
| `N B E` | `none_strict` | yes (none) | yes |
| four coords, then anything else before `E` | `unexpected_token` | no | no |
| four coords, no `E` (hit the token cap) | `incomplete` | no | no |
| prefix is not four coordinates | `bad_prefix` | no | no |
| tokens after `E` / duplicated `E` | `trailing_after_end` | no | no |
| non-finite or non-positive-area box | `degenerate_box` | no | no |

`historical_extractable` (report-only) = first four tokens are coordinates **and** the last
token is `E`; it exists to compare against older reports that ignored trailing tokens.

Geometry: the box must be finite with `x2>x1, y2>y1` after mapping back to the original image.
Boxes that fall outside the image are flagged `out_of_image` but **not clipped** (clipping would
silently change IoU).

## Why the "deployment valid" rule exists

Training labels are `C C C C B E`, but the trained model never emits `B` in free generation
(0/64, both on the board and in the float reference). A deployment parser therefore has to
accept `C C C C E` explicitly. This is a **versioned deployment policy**, not proof that the
model learned the original protocol. Never write a missing `B` into the raw token log.

## CLI and exit codes (student app)

```
student_v1 <asset-dir> one   <jpg> "<query>"      # single request
student_v1 <asset-dir> bench <manifest> <out.tsv> # one row per submitted request
student_v1 <asset-dir> ids   "<query>"            # prompt token ids only
student_v1 <asset-dir> pre|vraw|pix|emb ...       # diagnostics
```

| Exit | Meaning |
| ---: | --- |
| 0 | completed and deployment-valid |
| 3 | model/asset missing or init failed |
| 4 | runtime failure |
| 5 | parse invalid (status printed) |
| 6 | bench saw runtime failures or a malformed manifest |

`bench` always emits one row per input (`expected / submitted / completed / runtime_failed /
parse_invalid / malformed` in the summary line), so failures cannot silently shrink the
denominator.
