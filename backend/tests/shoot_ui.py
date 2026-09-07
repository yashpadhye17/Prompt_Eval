"""Screenshot the dashboard for visual verification.

Usage: .venv/bin/python backend/tests/shoot_ui.py [outdir]
Requires the backend (port 8000) and vite dev server (port 5173) to be running.
"""

from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/ui")
OUT.mkdir(parents=True, exist_ok=True)


def main() -> int:
    errors: list[str] = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1500, "height": 1100})
        page.on("console", lambda m: errors.append(f"{m.type}: {m.text}")
                if m.type == "error" else None)
        page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))

        page.goto("http://localhost:5173/", wait_until="networkidle")
        page.wait_for_timeout(2500)
        page.screenshot(path=OUT / "01-dashboard.png", full_page=True)
        print("captured dashboard")

        # Open the drill-down drawer from the first clickable cell row.
        rows = page.locator("tr.clickable")
        if rows.count() > 0:
            rows.first.click()
            page.wait_for_timeout(2200)
            page.screenshot(path=OUT / "02-drawer-response.png")
            print("captured drawer (response tab)")

            for label, name in [
                ("Numbers", "03-drawer-numbers.png"),
                ("Judge", "04-drawer-judge.png"),
            ]:
                btn = page.locator(f"button:has-text('{label}')").first
                if btn.count() > 0:
                    btn.click()
                    page.wait_for_timeout(900)
                    page.screenshot(path=OUT / name)
                    print(f"captured drawer ({label})")
        else:
            print("no result rows found to click")

        browser.close()

    if errors:
        print(f"\n{len(errors)} console error(s):")
        for e in errors[:20]:
            print("  -", e)
        return 1
    print("\nno console errors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
