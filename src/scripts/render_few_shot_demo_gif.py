"""Render a short looping GIF of the few-shot LLM flow (schematic).

Requires: pip install pillow

Usage:
    python -m src.scripts.render_few_shot_demo_gif

Writes: demo/few_shot_demo.gif
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_OUT = _REPO_ROOT / "demo" / "few_shot_demo.gif"

_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
)


def _load_font(size: int):
    from PIL import ImageFont

    for path in _FONT_CANDIDATES:
        p = Path(path)
        if p.is_file():
            try:
                return ImageFont.truetype(str(p), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def main() -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("Install: pip install pillow", file=sys.stderr)
        sys.exit(1)

    _OUT.parent.mkdir(parents=True, exist_ok=True)

    w, h = 800, 450
    bg = "#0f1419"
    accent = "#3d9ee5"
    text_c = "#c8d4e6"
    muted = "#8b9cb3"

    font_title = _load_font(22)
    font_body = _load_font(15)
    font_sub = _load_font(13)

    scenes = [
        (
            "1  Train split",
            "sample 3× Yes + 3× No from train.csv\n(random_balanced, fixed seed)",
            "exemplar_selector.select_exemplars",
        ),
        (
            "2  Format demonstrations",
            "Case 1 … Outcome: Yes\n…\nCase 6 … Outcome: No",
            "format_exemplar_block → demonstration_cases",
        ),
        (
            "3  Each test row",
            "predictor_few_shot.txt\n+ fixed 6 cases + this narrative_current",
            "run_predictions — one LLM call per pair_id",
        ),
        (
            "4  Parse & score",
            "Prediction: Yes / No  →  compare to label_lipid_disorder",
            "no fine-tuning — in-context only",
        ),
    ]

    frames: list = []
    hold = 10
    for title, body, sub in scenes:
        for f in range(hold):
            alpha = min(1.0, (f + 1) / 4.0)
            img = Image.new("RGB", (w, h), bg)
            draw = ImageDraw.Draw(img)
            draw.text((w // 2, 52), title, fill=accent, font=font_title, anchor="mm")
            # Multiline body
            y0 = 115
            for line in body.split("\n"):
                c = _blend_color(text_c, bg, alpha)
                draw.text((w // 2, y0), line, fill=c, font=font_body, anchor="mm")
                y0 += 26
            cs = _blend_color(muted, bg, 0.5 + 0.5 * alpha)
            draw.text((w // 2, h - 72), sub, fill=cs, font=font_sub, anchor="mm")
            frames.append(img)

    frames[0].save(
        _OUT,
        save_all=True,
        append_images=frames[1:],
        duration=120,
        loop=0,
        optimize=False,
    )
    print(f"Wrote {_OUT}")


def _blend_color(fg_hex: str, bg_hex: str, t: float) -> str:
    def hx(s: str) -> tuple[int, int, int]:
        s = s.lstrip("#")
        return tuple(int(s[i : i + 2], 16) for i in (0, 2, 4))

    a, b = hx(fg_hex), hx(bg_hex)
    t = max(0.0, min(1.0, t))
    r = int(a[0] * t + b[0] * (1 - t))
    g = int(a[1] * t + b[1] * (1 - t))
    bl = int(a[2] * t + b[2] * (1 - t))
    return f"#{r:02x}{g:02x}{bl:02x}"


if __name__ == "__main__":
    main()
