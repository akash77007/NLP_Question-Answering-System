"""
scripts/capture_screenshots.py
Captures high-resolution screenshots of the Streamlit Material You application.
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
    print("[*] Launching Streamlit web application on port 8501...")
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
            print("[-] Error: Streamlit did not start within 30 seconds.")
            return

        print("[*] Launching Chrome via Playwright...")
        with sync_playwright() as p:
            browser = p.chromium.launch(
                executable_path=CHROME_PATH,
                headless=True,
                args=["--no-sandbox", "--disable-gpu", "--window-size=1600,1100"]
            )
            context = browser.new_context(viewport={"width": 1600, "height": 1100})
            page = context.new_page()

            print("[*] Navigating to http://localhost:8501...")
            page.goto("http://localhost:8501")
            
            # Allow Streamlit to finish BERT load and initial render
            print("[*] Waiting for Streamlit React DOM to hydrate...")
            time.sleep(10)

            # 1. Initial Studio view (with preset dropdown & hero)
            shot1 = os.path.join(OUT_DIR, "01_ui_hero_studio.png")
            page.screenshot(path=shot1, full_page=False)
            print(f"[+] Saved: {shot1}")

            # 2. Select Apollo 11 preset
            print("[*] Selecting Apollo 11 preset...")
            try:
                page.locator('[data-testid="stSelectbox"]').first.click()
                time.sleep(0.8)
                page.locator('[role="option"]:has-text("Apollo 11")').first.click()
                time.sleep(1.2)
            except Exception as e:
                print(f"[-] Apollo selection: {e}")

            # Click primary extract button
            print("[*] Extracting answer for Apollo 11...")
            page.locator('button[kind="primary"]').first.click()
            time.sleep(4)

            shot2 = os.path.join(OUT_DIR, "02_answer_extracted_apollo.png")
            page.screenshot(path=shot2, full_page=False)
            print(f"[+] Saved: {shot2}")

            # 3. Expand diagnostics
            print("[*] Expanding Diagnostics...")
            try:
                page.locator('details summary').first.click()
                time.sleep(1)
                shot3 = os.path.join(OUT_DIR, "04_diagnostics_expanded.png")
                page.screenshot(path=shot3, full_page=False)
                print(f"[+] Saved: {shot3}")
            except Exception as e:
                print(f"[-] Diagnostics error: {e}")

            # 4. Select Eiffel Tower adversarial preset
            print("[*] Testing Adversarial Query (Eiffel Tower)...")
            try:
                page.locator('[data-testid="stSelectbox"]').first.click()
                time.sleep(0.8)
                page.locator('[role="option"]:has-text("Eiffel")').first.click()
                time.sleep(1.2)
                page.locator('button[kind="primary"]').first.click()
                time.sleep(4)
                shot4 = os.path.join(OUT_DIR, "03_adversarial_rejection_eiffel.png")
                page.screenshot(path=shot4, full_page=False)
                print(f"[+] Saved: {shot4}")
            except Exception as e:
                print(f"[-] Eiffel selection error: {e}")

            # 5. Tab 2: Architecture & Benchmark
            print("[*] Capturing Tab 2: Architecture & Benchmark...")
            try:
                page.locator('[data-baseweb="tab"]').nth(1).click()
                time.sleep(2)
                shot5 = os.path.join(OUT_DIR, "05_architecture_benchmark.png")
                page.screenshot(path=shot5, full_page=False)
                print(f"[+] Saved: {shot5}")
            except Exception as e:
                print(f"[-] Tab 2 error: {e}")

            # 6. Tab 3: Dataset EDA
            print("[*] Capturing Tab 3: Dataset EDA...")
            try:
                page.locator('[data-baseweb="tab"]').nth(2).click()
                time.sleep(2)
                shot6 = os.path.join(OUT_DIR, "06_dataset_eda.png")
                page.screenshot(path=shot6, full_page=False)
                print(f"[+] Saved: {shot6}")
            except Exception as e:
                print(f"[-] Tab 3 error: {e}")

            # 7. Tab 4: Group 10 Team
            print("[*] Capturing Tab 4: Group 10 Team...")
            try:
                page.locator('[data-baseweb="tab"]').nth(3).click()
                time.sleep(2)
                shot7 = os.path.join(OUT_DIR, "07_group10_team.png")
                page.screenshot(path=shot7, full_page=False)
                print(f"[+] Saved: {shot7}")
            except Exception as e:
                print(f"[-] Tab 4 error: {e}")

            browser.close()

    finally:
        print("[*] Shutting down Streamlit process...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
        print("[+] Streamlit stopped.")

if __name__ == "__main__":
    main()
