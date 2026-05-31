"""
01_scrape_novels.py — NovelFire Batch Scraper
==============================================
Usage:
    python 01_scrape_novels.py

Reads novel URLs from novels.txt (one URL per line, # for comments).
Scrapes up to 3 novels concurrently in headless mode.
Resumes automatically using scrape_progress.json.
Auto-pushes scraped data to git every 30 minutes.

Requirements:
    pip install playwright playwright-stealth beautifulsoup4 tqdm
    playwright install chromium
"""

import re
import queue
import time
import random
import json
import threading
import subprocess
from pathlib import Path
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from bs4 import BeautifulSoup

try:
    from playwright_stealth import stealth_sync
    _STEALTH_AVAILABLE = True
except ImportError:
    _STEALTH_AVAILABLE = False
    print("[WARN] playwright-stealth not installed. Run: pip install playwright-stealth")
    print("       Continuing without stealth mode (higher Cloudflare detection risk).")

try:
    from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
except ImportError:
    print("\n[ERROR] Playwright is not installed.")
    print("Run: pip install playwright && playwright install chromium\n")
    raise SystemExit(1)

OUTPUT_DIR   = Path("raw_novels")
PROGRESS_FILE = Path("scrape_progress.json")
NOVELS_FILE  = Path("novels.txt")
WORKERS          = 3    # Concurrent novels at once
GIT_PUSH_INTERVAL = 30  # Minutes between auto git pushes (0 to disable)

OUTPUT_DIR.mkdir(exist_ok=True)

# Lock so threads don't corrupt the shared progress dict / file
_progress_lock = threading.Lock()
_stop_event    = threading.Event()  # Signals background threads to stop

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def slugify(url: str) -> str:
    m = re.search(r"/book/([^/]+)", url)
    return m.group(1) if m else url.split("/")[-1]

def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {}

def save_progress(progress: dict):
    """Thread-safe progress save."""
    with _progress_lock:
        with open(PROGRESS_FILE, "w") as f:
            json.dump(progress, f, indent=2)

def load_urls_from_file() -> list[str]:
    """Read novel URLs from novels.txt. Creates an example file if missing."""
    if not NOVELS_FILE.exists():
        NOVELS_FILE.write_text(
            "# novels.txt — Add one NovelFire URL per line\n"
            "# Lines starting with # are ignored\n"
            "#\n"
            "# Example:\n"
            "# https://novelfire.net/book/the-beast-tamer-clans-monster-overlord\n"
        )
        print(f"[INFO] Created {NOVELS_FILE} — add your URLs there and re-run.")
        raise SystemExit(0)

    urls, seen = [], set()
    with open(NOVELS_FILE) as f:
        for i, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "novelfire.net/book/" not in line:
                print(f"  [SKIP] Line {i}: not a NovelFire URL → {line}")
                continue
            if line in seen:
                print(f"  [SKIP] Line {i}: duplicate → {line}")
                continue
            seen.add(line)
            urls.append(line)
    return urls

def random_delay(lo=1.2, hi=3.5):
    time.sleep(random.uniform(lo, hi))

# ─────────────────────────────────────────────
# Git auto-push daemon
# ─────────────────────────────────────────────

def git_push(label: str = "auto"):
    """Stage, commit, and push all changes. Silent on failure."""
    try:
        subprocess.run(["git", "add", "raw_novels/", "scrape_progress.json"],
                       capture_output=True, timeout=30)
        result = subprocess.run(
            ["git", "commit", "-m", f"[scraper] {label} — {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}"],
            capture_output=True, timeout=30
        )
        if result.returncode == 0:  # Only push if there was something new to commit
            subprocess.run(["git", "push"], capture_output=True, timeout=60)
            print(f"\n[GIT] Pushed: {label}")
    except Exception as e:
        print(f"\n[GIT] Push failed (non-fatal): {e}")

def git_push_loop():
    """Daemon thread: push to git every GIT_PUSH_INTERVAL minutes."""
    if GIT_PUSH_INTERVAL <= 0:
        return
    interval_secs = GIT_PUSH_INTERVAL * 60
    while not _stop_event.wait(timeout=interval_secs):
        git_push("periodic save")

# ─────────────────────────────────────────────
# Scraping core
# ─────────────────────────────────────────────

def wait_for_cloudflare(page, timeout=20_000):
    try:
        page.wait_for_function(
            "() => !document.title.includes('Just a moment')",
            timeout=timeout
        )
    except PWTimeout:
        pass  # headless can't solve CAPTCHAs; just carry on

def chapter_num(url: str) -> float:
    m = re.search(r'/chapter-(\d+(?:\.\d+)?)', url)
    return float(m.group(1)) if m else 0.0

def get_chapter_urls(page, novel_url: str) -> list[dict]:
    """Crawl the /chapters index (with pagination) to get full chapter list."""
    base = re.sub(r"/chapter-\d+.*$", "", novel_url.rstrip("/"))
    base = re.sub(r"/chapters/?.*$", "", base)
    m = re.search(r"/book/([^/]+)", base)
    if not m:
        raise ValueError(f"Cannot parse slug from: {novel_url}")
    slug = m.group(1)

    seen, chapters_dict = set(), {}
    current = base + "/chapters"

    while current:
        page.goto(current, wait_until="domcontentloaded", timeout=60_000)
        page.wait_for_timeout(2000)
        soup = BeautifulSoup(page.content(), "html.parser")

        for a in soup.find_all("a", href=True):
            abs_url = urljoin(page.url, a["href"])
            if f"/book/{slug}/chapter-" in abs_url and "/chapters" not in abs_url:
                if abs_url not in seen:
                    title = " ".join(a.get_text(separator=" ", strip=True).split())
                    title = re.sub(r'^\d+\s+', '', title)
                    title = re.sub(
                        r'\s+\d+\s+(second|minute|hour|day|week|month|year)s?\s+ago\s*$',
                        '', title, flags=re.IGNORECASE
                    )
                    seen.add(abs_url)
                    chapters_dict[abs_url] = title

        # Advance pagination
        nxt = soup.find("a", rel="next")
        if nxt and nxt.get("href"):
            n = urljoin(page.url, nxt["href"])
            current = n if n != current else None
        else:
            pg = soup.find("ul", class_="pagination")
            n = None
            if pg:
                for a in pg.find_all("a", href=True):
                    if "›" in a.get_text() or "next" in a.get_text().lower():
                        n = urljoin(page.url, a["href"])
                        break
            current = n if n and n != current else None

    result = [{"title": t, "url": u} for u, t in chapters_dict.items()]
    result.sort(key=lambda x: chapter_num(x["url"]))
    return result


def scrape_chapter_text(page, url: str, retries=5) -> str | None:
    """Load a chapter page and return clean paragraph text."""
    for attempt in range(1, retries + 1):
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=60_000)
            wait_for_cloudflare(page)
            page.wait_for_selector("#content", timeout=15_000)

            soup = BeautifulSoup(page.content(), "html.parser")
            div = soup.find("div", id="content")
            if not div:
                return None

            for tag in div.find_all(["iframe", "script", "style", "ins"]):
                tag.decompose()

            paras = [p.get_text(separator=" ", strip=True) for p in div.find_all("p")]
            paras = [p for p in paras if p]
            return "\n\n".join(paras) if paras else None

        except PWTimeout:
            if attempt < retries:
                random_delay(3, 6)
        except Exception:
            if attempt < retries:
                random_delay(3, 6)
    return None


# ─────────────────────────────────────────────
# Per-novel scrape (called by the queue worker)
# ─────────────────────────────────────────────

def scrape_one_novel(page, novel_url: str, progress: dict) -> tuple[str, bool]:
    """
    Scrape a single novel on the given page.
    Returns (slug, success).
    Raises RuntimeError on chapter failure so the queue worker can move on.
    """
    slug = slugify(novel_url)
    output_file = OUTPUT_DIR / f"{slug}.txt"
    tag = f"[{slug[:35]}]"

    print(f"{tag} Discovering chapters...")
    chapters = get_chapter_urls(page, novel_url)
    print(f"{tag} Found {len(chapters)} chapters.")

    with _progress_lock:
        novel_progress = progress.get(slug, {"done": [], "failed": []})

    done_set = set(novel_progress["done"])
    remaining = [c for c in chapters if c["url"] not in done_set]

    if not remaining:
        print(f"{tag} Already complete — skipping.")
        return slug, True

    print(f"{tag} {len(done_set)} done, {len(remaining)} to go.")

    with open(output_file, "a", encoding="utf-8") as f:
        for ch in tqdm(remaining, desc=f"  {slug[:38]}", unit="ch"):
            text = scrape_chapter_text(page, ch["url"], retries=5)

            if text:
                f.write(f"\n\n{'─'*60}\n{ch['title']}\n{'─'*60}\n\n")
                f.write(text)
                f.flush()
                with _progress_lock:
                    novel_progress["done"].append(ch["url"])
                    if ch["url"] in novel_progress.get("failed", []):
                        novel_progress["failed"].remove(ch["url"])
                    progress[slug] = novel_progress
                save_progress(progress)
            else:
                # Log the failure and stop this novel to preserve chapter order.
                # The queue worker will immediately pick up the next novel.
                print(f"\n{tag} [FAIL] {ch['title']} — stopping to preserve order.")
                with _progress_lock:
                    if ch["url"] not in novel_progress.get("failed", []):
                        novel_progress.setdefault("failed", []).append(ch["url"])
                    progress[slug] = novel_progress
                save_progress(progress)
                return slug, False  # signal failure without crashing the worker

            random_delay()

    failed_count = len(novel_progress.get("failed", []))
    print(f"\n{tag} ✓ Saved → {output_file}  (failed chapters: {failed_count})")
    return slug, True


# ─────────────────────────────────────────────
# Queue-draining worker thread
# ─────────────────────────────────────────────

def queue_worker(novel_queue: queue.Queue, progress: dict, results: dict):
    """
    Each thread runs this loop: pull a novel URL from the queue,
    scrape it (stopping cleanly on failure), then pull the next one.
    Exits only when the queue is empty.
    """
    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            headless=True,
            args=["--disable-blink-features=AutomationControlled"]
        )
        ctx = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 800},
            locale="en-US",
        )
        ctx.route("**/*.{png,jpg,gif,svg,woff,woff2,ttf}", lambda r: r.abort())
        ctx.route("**/ad.a-ads.com/**",         lambda r: r.abort())
        ctx.route("**/googletagmanager.com/**", lambda r: r.abort())

        page = ctx.new_page()
        if _STEALTH_AVAILABLE:
            stealth_sync(page)  # Patch all headless detection vectors

        try:
            while True:
                try:
                    novel_url = novel_queue.get_nowait()
                except queue.Empty:
                    break  # no more novels — this worker is done

                try:
                    slug, ok = scrape_one_novel(page, novel_url, progress)
                    with _progress_lock:
                        if ok:
                            results["ok"].append(slug)
                        else:
                            results["failed"].append(slug)
                except Exception as e:
                    slug = slugify(novel_url)
                    print(f"\n[{slug[:35]}] Unexpected error: {e}")
                    with _progress_lock:
                        results["failed"].append(slug)
                finally:
                    novel_queue.task_done()
        finally:
            browser.close()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print(f"  NovelFire Batch Scraper  —  {WORKERS} concurrent workers")
    print("="*60)

    urls = load_urls_from_file()
    if not urls:
        print(f"[EXIT] No valid URLs in {NOVELS_FILE}.")
        return

    progress = load_progress()
    print(f"\n[→] {len(urls)} novel(s) queued from {NOVELS_FILE}")
    print(f"[→] Headless mode, {WORKERS} workers\n")

    # Fill the shared queue
    novel_queue: queue.Queue = queue.Queue()
    for u in urls:
        novel_queue.put(u)

    results = {"ok": [], "failed": []}

    # Start background git-push daemon
    if GIT_PUSH_INTERVAL > 0:
        git_thread = threading.Thread(target=git_push_loop, daemon=True)
        git_thread.start()
        print(f"[→] Auto git-push every {GIT_PUSH_INTERVAL} min\n")

    try:
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            # Each thread is a persistent worker that drains the queue
            futures = [pool.submit(queue_worker, novel_queue, progress, results)
                       for _ in range(min(WORKERS, len(urls)))]
            for f in futures:
                f.result()  # propagate any unhandled thread exceptions
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Progress saved — re-run to continue.")
    finally:
        _stop_event.set()  # Stop the git push loop
        git_push("final save")  # One last push when everything is done

    ok, failed = results["ok"], results["failed"]
    print("\n" + "="*60)
    print(f"  {len(ok)} succeeded  |  {len(failed)} stopped early")
    if failed:
        print("  Stopped early (will resume on next run):")
        for s in failed:
            print(f"    ✗ {s}")
    print(f"  Output: {OUTPUT_DIR.resolve()}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
