"""Capture real Web UI screenshots for the report figures (Playwright + Edge)."""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = "http://127.0.0.1:8000"
OUT = Path("docs/figures")
PDF_DOC_ID = "P020230905354801948345-73262d2774f8"


def shot(page, selector: str, name: str) -> None:
    """Screenshot an element via its bounding box (robust to the 5s refresh)."""
    try:
        page.locator(selector).scroll_into_view_if_needed(timeout=3000)
    except Exception:
        pass
    page.wait_for_timeout(700)
    try:
        box = page.locator(selector).bounding_box()
        if box and box["width"] > 1 and box["height"] > 1:
            page.screenshot(path=str(OUT / name), clip=box)
            return
    except Exception:
        pass
    page.screenshot(path=str(OUT / name))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 960},
                                device_scale_factor=2)
        page.goto(BASE, wait_until="networkidle")
        page.wait_for_function("document.getElementById('snapshot').options.length > 0")
        page.wait_for_function("document.getElementById('admin').hidden === false", timeout=20000)

        # 1. Model & key settings panel (bottom of the left rail).
        page.click("#admin summary")
        page.wait_for_function("document.getElementById('models').children.length > 0", timeout=20000)
        shot(page, "#admin", "model_settings.png")

        # 2. Evidence traceability: retrieval Top 5 with sources.
        page.select_option("#docFilter", PDF_DOC_ID)
        page.uncheck("#answer")
        page.fill("#question", "河南安阳市凯信达商贸有限公司“11·21”特别重大火灾事故的直接原因是什么？")
        page.click("#ask")
        try:
            page.wait_for_function("document.getElementById('hits').children.length > 0", timeout=60000)
        except Exception:
            page.wait_for_timeout(3000)
        page.wait_for_timeout(1500)
        shot(page, "#hits", "evidence_hits.png")

        # 3. Product feedback buttons + usage statistics.
        try:
            page.wait_for_function("document.getElementById('feedback').hidden === false", timeout=20000)
        except Exception:
            pass
        shot(page, "#feedback", "feedback_buttons.png")
        page.wait_for_function("document.getElementById('metrics').textContent.trim().length > 3", timeout=20000)
        shot(page, "#metrics", "metrics.png")

        # Print DOM text for verification (no secrets).
        print("MODELS_TEXT:", page.eval_on_selector("#models", "el => el.innerText"))
        print("HITS_TEXT:", page.eval_on_selector("#hits", "el => el.innerText")[:400])
        print("FEEDBACK_TEXT:", page.eval_on_selector("#feedback", "el => el.innerText"))
        print("METRICS_TEXT:", page.eval_on_selector("#metrics", "el => el.innerText"))

        page.screenshot(path=str(OUT / "full_page.png"), full_page=True)
        browser.close()
    print("captured:", sorted(x.name for x in OUT.iterdir() if x.suffix == ".png"))


if __name__ == "__main__":
    sys.exit(main())
