"""Record an end-to-end demo video of the running app with Playwright (no narration; on-screen captions).

    python run_app.py                 # in one terminal
    python scripts/record_demo.py     # writes outputs/demo/demo.webm (+ step screenshots)

Requires the dev extras: pip install -r requirements-dev.txt && python -m playwright install chromium
"""
from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUTD = ROOT / "outputs" / "demo"
URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8501"


def caption(page, text):
    page.evaluate("""t => {let d=document.getElementById('demo-cap');
        if(!d){d=document.createElement('div');d.id='demo-cap';
        d.style.cssText='position:fixed;bottom:18px;left:50%;transform:translateX(-50%);z-index:99999;'+
        'background:rgba(0,0,0,.82);color:#fff;font:600 18px sans-serif;padding:10px 18px;border-radius:8px;max-width:80%';
        document.body.appendChild(d);} d.textContent=t;}""", text)


def settle(page, s=2.5):
    time.sleep(0.5)
    try:
        page.wait_for_selector("[data-testid='stStatusWidget']", state="detached", timeout=180000)
    except Exception:
        pass
    time.sleep(s)


def click_map(page, fx, fy):
    """Click at a fraction of the map image (rendered inside the image-coordinates iframe)."""
    fr = page.frame_locator("iframe[title*='streamlit_image_coordinates']").first
    img = fr.locator("img").first
    box = img.bounding_box()
    if box is None:  # fallback: iframe element box
        box = page.locator("iframe[title*='streamlit_image_coordinates']").first.bounding_box()
    page.mouse.click(box["x"] + fx * box["width"], box["y"] + fy * box["height"])


def select_scene(page, label):
    page.locator("[data-testid='stSidebar'] [data-testid='stSelectbox']").first.click()
    page.get_by_role("option", name=label).click()


def main():
    OUTD.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        br = p.chromium.launch()
        ctx = br.new_context(viewport={"width": 1440, "height": 900}, record_video_dir=str(OUTD / "raw"),
                             record_video_size={"width": 1440, "height": 900})
        page = ctx.new_page()
        t0 = time.time()
        page.goto(URL)
        settle(page, 4)
        shots = 0

        def step(text, pause=4.0):
            nonlocal shots
            caption(page, text)
            time.sleep(pause)
            shots += 1
            page.screenshot(path=str(OUTD / f"step{shots:02d}.png"))

        step("Spectral Earth Mini: Indian Pines, 3-D CNN trained on the leakage-free spatial split", 5)
        click_map(page, 0.45, 0.55)
        settle(page)
        step("Click a pixel: spectrum, predicted class, confidence, top-3 and ground truth", 5)
        page.get_by_text("RBF-SVM (spectral)").click()
        settle(page)
        step("Switch model: RBF-SVM full-scene inference from the saved model", 4)
        page.get_by_text("Confidence", exact=True).first.click()
        settle(page)
        step("Toggle the confidence layer (temperature-scaled / Platt probabilities)", 4)
        page.get_by_text("Index", exact=True).first.click()
        settle(page)
        step("Index layer: NDVI on predicted crop classes; opacity sliders blend every layer", 4)
        page.mouse.wheel(0, 1400)
        time.sleep(1.5)
        step("Summary: class areas, mean confidence, spatial vs random metrics; PNG / TIFF / PDF export", 6)
        page.mouse.wheel(0, -3000)
        select_scene(page, "Salinas")
        settle(page, 4)
        step("Salinas: same pipeline transferred unchanged (CNN inference on 111k pixels)", 5)
        click_map(page, 0.5, 0.35)
        settle(page)
        step("Pixel inspection on Salinas", 4)
        select_scene(page, "Cuprite")
        settle(page, 4)
        step("Cuprite: SAM mineral map on continuum-removed SWIR (no ground truth - no accuracy claimed)", 5)
        click_map(page, 0.86, 0.3)
        settle(page)
        step("Pixel vs matched USGS library spectrum: SAM angle, MLP probability, feature-fit residual", 6)
        page.get_by_text("MLP (synthetic-trained)").click()
        settle(page)
        step("Synthetic-trained MLP with an unknown/none class", 4)
        page.get_by_text("Re-run MLP inference with saved model").click()
        settle(page, 3)
        step("Live inference with the saved MLP reproduces the stored map", 5)
        page.mouse.wheel(0, 1400)
        time.sleep(1.5)
        step("Validation evidence: SAM-vs-MLP agreement; exports for the mineral layers", 6)
        caption(page, f"End of demo ({time.time() - t0:.0f} s)")
        time.sleep(2)
        ctx.close()
        br.close()
    vids = sorted((OUTD / "raw").glob("*.webm"))
    if vids:
        shutil.move(str(vids[-1]), OUTD / "demo.webm")
        shutil.rmtree(OUTD / "raw", ignore_errors=True)
        print("wrote", OUTD / "demo.webm")


if __name__ == "__main__":
    main()
