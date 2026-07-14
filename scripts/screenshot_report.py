"""Capture docs/assets/report.png from the demo report (run the notebook first).

Regenerate: uv run --group demo playwright install chromium
            uv run --group demo python scripts/screenshot_report.py
"""

from pathlib import Path

from playwright.sync_api import sync_playwright

html = Path("notebooks/golden_cross_report.html").resolve()
out = Path("docs/assets/report.png")
with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1000, "height": 1400}, device_scale_factor=2)
    page.goto(html.as_uri())
    page.wait_for_timeout(2000)  # let plotly render
    page.screenshot(path=str(out), clip={"x": 0, "y": 0, "width": 1000, "height": 1400})
    browser.close()
print(f"wrote {out}")
