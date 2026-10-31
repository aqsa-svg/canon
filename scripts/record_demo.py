"""Record the demo driving itself, as frames, then assemble them with ffmpeg.

**This records the real application.** It starts nothing and fakes nothing: it drives a
running Streamlit server with a real browser, clicks the real buttons and screenshots
whatever actually appears. A hand-made animation of what the app *would* do would be a
claim about behaviour nobody could check -- the precise thing this project argues
against -- so the recording is only worth having if it is genuinely a recording.

It costs no tokens. Every example replays from ``demo_cache/``, which is the property
that makes the public demo safe and also makes it recordable on demand.

Usage:
    python -m streamlit run app.py --server.headless true --server.port 8599
    python scripts/record_demo.py --url http://localhost:8599
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FRAMES_DIR = REPO_ROOT / "docs" / "_frames"
MP4_PATH = REPO_ROOT / "docs" / "demo.mp4"
GIF_PATH = REPO_ROOT / "docs" / "demo.gif"

VIEWPORT = {"width": 1280, "height": 800}
FPS = 12
"""Frames per second in the output. The capture rate is lower -- each screenshot is held
for several frames -- because a demo is read, not watched: a viewer needs time to see the
SQL, not a smooth animation of it appearing."""


def _dismiss_overlays(page: object) -> None:
    """Close any Streamlit dialog or toast that would swallow the next click.

    One appeared mid-run and intercepted pointer events, which failed the recording
    rather than producing a wrong frame -- but it failed it slowly, through retries.
    Escape plus an explicit removal is cheaper than diagnosing it again.
    """
    try:
        page.keyboard.press("Escape")  # type: ignore[attr-defined]
        page.evaluate(  # type: ignore[attr-defined]
            """() => document.querySelectorAll('[data-testid=\"stDialog\"]')
                     .forEach(node => node.remove())"""
        )
    except Exception as error:
        print(f"  (overlay dismissal skipped: {type(error).__name__})")


def _hold(page: object, out: Path, index: int, seconds: float) -> int:
    """Screenshot once and repeat the frame for ``seconds``. Returns the next index."""
    shot = out / f"frame_{index:04d}.png"
    page.screenshot(path=str(shot))  # type: ignore[attr-defined]
    for offset in range(1, int(seconds * FPS)):
        shutil.copyfile(shot, out / f"frame_{index + offset:04d}.png")
    return index + max(int(seconds * FPS), 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8599")
    parser.add_argument(
        "--examples",
        type=int,
        default=3,
        help="how many of the four examples to walk through",
    )
    args = parser.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print(
            "playwright is not installed. It is a recording tool, not a runtime "
            "dependency, so it is deliberately absent from requirements.txt. Install "
            "it with: pip install playwright && playwright install chromium"
        )
        return 1

    if FRAMES_DIR.exists():
        shutil.rmtree(FRAMES_DIR)
    FRAMES_DIR.mkdir(parents=True)

    index = 0
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport=VIEWPORT, device_scale_factor=2)
        page.goto(args.url, wait_until="networkidle", timeout=120_000)
        # Streamlit renders progressively; settle before the first frame rather than
        # capturing a half-drawn page.
        page.wait_for_selector("text=Canon", timeout=60_000)
        # The toolbar is Streamlit's chrome ("Deploy", the hamburger), not the app. It
        # is hidden for the recording only -- nothing in the page itself is altered.
        page.add_style_tag(
            content='[data-testid="stToolbar"], [data-testid="stDecoration"],'
            '[data-testid="stStatusWidget"] { display: none !important; }'
        )
        time.sleep(3)
        _dismiss_overlays(page)
        index = _hold(page, FRAMES_DIR, index, 3.0)

        buttons = page.get_by_role("button")
        labels = [
            label for label in buttons.all_inner_texts() if label.strip() and label.strip() != "Ask"
        ]
        print(f"examples found: {labels}")

        for label in labels[: args.examples]:
            _dismiss_overlays(page)
            page.get_by_role("button", name=label, exact=True).click()
            time.sleep(1.5)
            index = _hold(page, FRAMES_DIR, index, 1.2)

            _dismiss_overlays(page)
            page.get_by_role("button", name="Ask", exact=True).click()
            page.wait_for_timeout(4000)
            index = _hold(page, FRAMES_DIR, index, 3.5)

            # Scroll through the answer so the SQL and the metric card are both seen --
            # a demo of a system whose claim is "here is the query that produced it"
            # that never shows the query is worse than no demo.
            #
            # By mouse wheel, not window.scrollTo: Streamlit scrolls an inner container,
            # so scrolling `document.body` moved nothing and the first recording showed
            # only the top of every answer.
            page.mouse.move(VIEWPORT["width"] // 2, VIEWPORT["height"] // 2)
            for _ in range(3):
                page.mouse.wheel(0, 520)
                time.sleep(0.9)
                index = _hold(page, FRAMES_DIR, index, 2.0)
            page.mouse.wheel(0, -2600)
            time.sleep(0.8)

        browser.close()

    print(f"{index} frames in {FRAMES_DIR}")
    _assemble()
    return 0


def _assemble() -> None:
    """Frames to MP4 (for posting) and GIF (for the README), via ffmpeg."""
    pattern = str(FRAMES_DIR / "frame_%04d.png")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(FPS),
            "-i",
            pattern,
            # yuv420p and even dimensions: without both, the file plays nowhere.
            "-vf",
            "scale=1280:-2:flags=lanczos,format=yuv420p",
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "23",
            "-movflags",
            "+faststart",
            str(MP4_PATH),
        ],
        check=True,
    )
    palette = FRAMES_DIR / "palette.png"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(MP4_PATH),
            "-vf",
            "fps=8,scale=900:-1:flags=lanczos,palettegen=max_colors=128",
            str(palette),
        ],
        check=True,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(MP4_PATH),
            "-i",
            str(palette),
            "-lavfi",
            "fps=8,scale=900:-1:flags=lanczos[x];[x][1:v]paletteuse",
            "-loop",
            "0",
            str(GIF_PATH),
        ],
        check=True,
    )
    for path in (MP4_PATH, GIF_PATH):
        print(f"  {path.name}: {path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    raise SystemExit(main())
