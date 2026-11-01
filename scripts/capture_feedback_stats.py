"""Capture the post-query feedback + anonymous stats state as one figure."""
from playwright.sync_api import sync_playwright

BASE = "http://127.0.0.1:8000"
OUT = "docs/figures"
PDF_DOC_ID = "P020230905354801948345-73262d2774f8"


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 960}, device_scale_factor=2)
        page.goto(BASE, wait_until="networkidle")
        page.wait_for_function("document.getElementById('snapshot').options.length > 0")

        page.select_option("#docFilter", PDF_DOC_ID)
        page.uncheck("#answer")
        page.fill("#question", "河南安阳市凯信达商贸有限公司“11·21”特别重大火灾事故的直接原因是什么？")
        page.click("#ask")
        try:
            page.wait_for_function("document.getElementById('hits').children.length > 0", timeout=60000)
        except Exception:
            pass
        page.wait_for_timeout(1500)
        try:
            page.wait_for_function("document.getElementById('feedback').hidden === false", timeout=20000)
        except Exception:
            pass

        # Open the usage-statistics section so the anonymous counters are visible.
        page.locator("summary", has_text="使用统计").click()
        page.wait_for_timeout(700)
        page.screenshot(path=f"{OUT}/feedback_stats.png")
        browser.close()
    print("feedback_stats captured")


if __name__ == "__main__":
    main()
