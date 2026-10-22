#!/usr/bin/env python3
"""OPA / PayGuard smoke driver.

  smoke.py api  [--api URL] [--user NAME]            # health → login → cases → case detail
  smoke.py ui   [--web URL] [--user NAME] [--path P] [--out PNG]
                                                       # log in through the form, open a page, screenshot
  smoke.py all  (both)

Run with the shared venv's python (it has playwright + its Chromium):
  /Users/issam/claude/.venv/bin/python .claude/skills/run-opa/smoke.py all

Demo auth: password == username (auth_service.authenticate_user). Exit code is
non-zero on the first failed check, so it works as a gate.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _req(method: str, url: str, body: dict | None = None, token: str | None = None) -> tuple[int, object]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("content-type", "application/json")
    if token:
        req.add_header("authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            raw = r.read()
            return r.status, (json.loads(raw) if raw else None)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")[:300]


def check(ok: bool, msg: str) -> None:
    print(("  ok   " if ok else "  FAIL ") + msg)
    if not ok:
        sys.exit(1)


def api(base: str, user: str) -> None:
    print(f"[api] {base}")
    code, body = _req("GET", f"{base}/health")
    check(code == 200 and body.get("status") == "ok", f"GET /health -> {code} {body}")

    code, body = _req("POST", f"{base}/api/auth/login", {"username": user, "password": user})
    check(code == 200 and "access_token" in body, f"POST /api/auth/login as {user} -> {code}")
    token = body["access_token"]
    print(f"       role={body['role']} name={body['full_name']}")

    # No token is NOT rejected: get_current_user falls back to the `system` user
    # (demo identity, not auth — see CLAUDE.md "Auth & identity"). Report, don't gate.
    code, _ = _req("GET", f"{base}/api/cases")
    print(f"       (no-token GET /api/cases -> {code}; system-user fallback)")

    # NB: the list endpoint pages with page/page_size; `limit=` is silently ignored.
    code, body = _req("GET", f"{base}/api/cases?page=1&page_size=5", token=token)
    check(code == 200 and body["total"] > 0, f"GET /api/cases -> {code} total={body.get('total') if isinstance(body, dict) else body}")
    top = body["items"][0]
    print(f"       first: {top['case_number']} {top['status']} {top['priority']} "
          f"priority_score={top['priority_score']} evidence={top['evidence_score']}")

    # NB: the path param is *named* case_id but takes the integer `id`; the UUID `case_id` -> 422.
    code, detail = _req("GET", f"{base}/api/cases/{top['id']}", token=token)
    check(code == 200 and detail.get("case_number") == top["case_number"],
          f"GET /api/cases/{top['id']} -> {code} detector_results={len(detail.get('detector_results', [])) if isinstance(detail, dict) else '?'}")

    code, counts = _req("GET", f"{base}/api/cases/status-counts", token=token)
    check(code == 200 and isinstance(counts, dict), f"GET /api/cases/status-counts -> {counts}")


def _skip_tour(page) -> None:
    """A fresh browser profile gets the 10-step 'Analyst tour' modal over every
    page until dismissed; it blocks clicks on the table underneath."""
    skip = page.get_by_text("Skip", exact=True)
    if skip.count() and skip.first.is_visible():
        skip.first.click()
        page.wait_for_timeout(300)


def ui(web: str, user: str, path: str, out: Path) -> None:
    from playwright.sync_api import sync_playwright

    print(f"[ui] {web}{path}")
    errors: list[str] = []
    with sync_playwright() as p:
        # The venv's playwright pins a Chromium build that may not be in
        # ~/Library/Caches/ms-playwright; system Chrome sidesteps the download.
        try:
            browser = p.chromium.launch(headless=True, channel="chrome")
        except Exception:  # noqa: BLE001 — no system Chrome → bundled build
            browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.on("response", lambda r: r.status >= 400 and errors.append(f"{r.status} {r.request.method} {r.url}"))
        page.goto(f"{web}/login", wait_until="networkidle")
        page.fill('input[placeholder="e.g., alice.johnson"]', user)
        page.fill('input[placeholder="Enter your password"]', user)
        page.click('button[type="submit"]')
        page.wait_for_url(lambda u: "/login" not in u, timeout=15000)
        check(True, f"logged in as {user} -> {page.url}")

        page.goto(f"{web}{path}", wait_until="networkidle")
        page.wait_for_timeout(1500)  # charts/tables render after the fetches settle
        _skip_tour(page)
        out.parent.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(out), full_page=False)
        text = page.inner_text("body")
        check("OPA-" in text or path != "/worklist", f"{path} shows case numbers (OPA-...)")
        print(f"       screenshot: {out}")

        if path == "/worklist":
            # Click through to the first case — the core analyst flow.
            row = page.locator("text=/^OPA-(2026|CAP)-\\d+$/").first
            case_no = row.inner_text().strip()
            row.click()
            page.wait_for_url("**/cases/**", timeout=15000)
            page.wait_for_load_state("networkidle")
            page.wait_for_timeout(1000)
            _skip_tour(page)
            detail_png = out.with_name("case-detail.png")
            page.screenshot(path=str(detail_png), full_page=False)
            check(case_no in page.inner_text("body"), f"opened {case_no} -> {page.url}")
            print(f"       screenshot: {detail_png}")
        if errors:
            print(f"       failed requests ({len(errors)}): {errors[:5]}")
        browser.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["api", "ui", "all"])
    ap.add_argument("--api", default="http://localhost:8001")
    ap.add_argument("--web", default="http://localhost:5174")
    ap.add_argument("--user", default="ana.chen")
    ap.add_argument("--path", default="/worklist")
    ap.add_argument("--out", default=str(HERE / "screenshots" / "worklist.png"))
    a = ap.parse_args()
    if a.mode in ("api", "all"):
        api(a.api, a.user)
    if a.mode in ("ui", "all"):
        ui(a.web, a.user, a.path, Path(a.out))
    print("PASS")


if __name__ == "__main__":
    main()
