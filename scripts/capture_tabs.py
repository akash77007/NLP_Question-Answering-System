"""
scripts/capture_tabs.py
Captures:
- 05_architecture_benchmark.png (Tab 2)
- 06_dataset_eda.png (Tab 3)
- 07_group10_team.png (Tab 4)
"""

import os
import sys
import time
import subprocess
import urllib.request
from playwright.sync_api import sync_playwright

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
VENV_PYTHON = r"D:\Documents\Assignments\NLP\Question_Answering_System\venv\Scripts\python.exe"
BASE_DIR = r"D:\Documents\Assignments\NLP\Question_Answering_System"
OUT_DIR = os.path.join(BASE_DIR, "docs", "screenshots")

def wait_for_server(url="http://localhost:8501", timeout=30):
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url) as response:
                if response.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False

def main():
    print("[*] Launching Streamlit...", flush=True)
    proc = subprocess.Popen(
        [
            VENV_PYTHON, "-m", "streamlit", "run", "app.py",
            "--server.port=8501",
            "--server.headless=true",
            "--browser.gatherUsageStats=false"
        ],
        cwd=BASE_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    try:
        if not wait_for_server():
            print("[-] Error: Streamlit did not start.", flush=True)
            return

        print("[*] Launching Chrome...", flush=True)
        with sync_playwright() as p:
            browser = p.chromium.launch(
                executable_path=CHROME_PATH,
                headless=True,
                args=["--no-sandbox", "--disable-gpu", "--window-size=1600,1100"]
            )
            context = browser.new_context(viewport={"width": 1600, "height": 1100})
            page = context.new_page()

            page.goto("http://localhost:8501")
            print("[*] Waiting for stTabs container...", flush=True)
            page.wait_for_selector('[data-testid="stTabs"]', timeout=35000)
            time.sleep(2)

            tabs = page.locator('[data-testid="stTabs"] button').all()
            print(f"[*] Found {len(tabs)} tabs via stTabs button.", flush=True)

            if len(tabs) >= 2:
                print("[*] Clicking Tab 2...", flush=True)
                tabs[1].click()
                time.sleep(2)
                shot = os.path.join(OUT_DIR, "05_architecture_benchmark.png")
                page.screenshot(path=shot, full_page=False)
                print(f"[+] Saved: {shot}", flush=True)

            if len(tabs) >= 3:
                print("[*] Clicking Tab 3...", flush=True)
                tabs[2].click()
                time.sleep(2)
                shot = os.path.join(OUT_DIR, "06_dataset_eda.png")
                page.screenshot(path=shot, full_page=False)
                print(f"[+] Saved: {shot}", flush=True)

            if len(tabs) >= 4:
                print("[*] Clicking Tab 4...", flush=True)
                tabs[3].click()
                time.sleep(2)
                shot = os.path.join(OUT_DIR, "07_group10_team.png")
                page.screenshot(path=shot, full_page=False)
                print(f"[+] Saved: {shot}", flush=True)

            browser.close()

    finally:
        print("[*] Terminating Streamlit...", flush=True)
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
        print("[+] Streamlit terminated.", flush=True)

if __name__ == "__main__":
    main()
