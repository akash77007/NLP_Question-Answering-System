"""
scripts/capture_remaining_screenshots.py
Captures:
- 03_adversarial_rejection_eiffel.png (Adversarial query)
- 05_architecture_benchmark.png (Tab 2)
- 06_dataset_eda.png (Tab 3)
- 07_group10_team.png (Tab 4)
- 01_ui_hero_studio.png (Tab 1 Clean)
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

os.makedirs(OUT_DIR, exist_ok=True)

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
    print("[*] Launching Streamlit on port 8501...", flush=True)
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
            print("[-] Error: Streamlit did not start within 30s.", flush=True)
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

            print("[*] Navigating to http://localhost:8501...", flush=True)
            page.goto("http://localhost:8501")
            
            print("[*] Waiting for DOM hydration...", flush=True)
            page.wait_for_selector("textarea", timeout=25000)
            time.sleep(3)

            # 1. Clean hero screenshot
            shot1 = os.path.join(OUT_DIR, "01_ui_hero_studio.png")
            page.screenshot(path=shot1, full_page=False)
            print(f"[+] Saved: {shot1}", flush=True)

            # 2. Adversarial query test
            print("[*] Direct filling Adversarial Query (Eiffel Tower)...", flush=True)
            ctx = ("The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. "
                   "It was constructed from 1887 to 1889 as the centerpiece of the 1889 World's Fair. "
                   "Named after the engineer Gustave Eiffel, it is 330 metres tall.")
            q = "How many kilograms of gold were used to paint the Eiffel Tower?"
            
            page.locator("textarea").first.fill(ctx)
            time.sleep(0.5)
            page.locator('input[type="text"]').first.fill(q)
            time.sleep(0.5)
            
            page.locator('button[kind="primary"]').first.click()
            time.sleep(3.5)

            shot_adv = os.path.join(OUT_DIR, "03_adversarial_rejection_eiffel.png")
            page.screenshot(path=shot_adv, full_page=False)
            print(f"[+] Saved: {shot_adv}", flush=True)

            # 3. Tab Navigation
            tabs = page.locator('[data-baseweb="tab"]').all()
            print(f"[*] Found {len(tabs)} tabs.", flush=True)

            if len(tabs) >= 2:
                print("[*] Clicking Tab 2 (Architecture)...", flush=True)
                tabs[1].click()
                time.sleep(2)
                shot_tab2 = os.path.join(OUT_DIR, "05_architecture_benchmark.png")
                page.screenshot(path=shot_tab2, full_page=False)
                print(f"[+] Saved: {shot_tab2}", flush=True)

            if len(tabs) >= 3:
                print("[*] Clicking Tab 3 (Dataset EDA)...", flush=True)
                tabs[2].click()
                time.sleep(2)
                shot_tab3 = os.path.join(OUT_DIR, "06_dataset_eda.png")
                page.screenshot(path=shot_tab3, full_page=False)
                print(f"[+] Saved: {shot_tab3}", flush=True)

            if len(tabs) >= 4:
                print("[*] Clicking Tab 4 (Group 10 Team)...", flush=True)
                tabs[3].click()
                time.sleep(2)
                shot_tab4 = os.path.join(OUT_DIR, "07_group10_team.png")
                page.screenshot(path=shot_tab4, full_page=False)
                print(f"[+] Saved: {shot_tab4}", flush=True)

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
