"""
docs/build_pdf_report.py
Converts docs/Project_Report.md into a publication-quality academic PDF report (A4).
Uses markdown, MathJax/CSS math styling, and Chrome's headless print-to-pdf engine.
"""

import os
import sys
import re
import subprocess
import markdown

BASE_DIR = r"D:\Documents\Assignments\NLP\Question_Answering_System"
CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
REPORT_MD = os.path.join(BASE_DIR, "docs", "Project_Report.md")
REPORT_HTML = os.path.join(BASE_DIR, "docs", "Project_Report.html")
REPORT_PDF = os.path.join(BASE_DIR, "docs", "Project_Report.pdf")

def generate_pdf_report():
    print(f"[*] Reading markdown report from: {REPORT_MD}")
    with open(REPORT_MD, "r", encoding="utf-8") as f:
        md_text = f.read()

    # Pre-process image paths so Chrome can resolve them accurately
    # 1. ../data/squad_eda_plots.png -> absolute file URI or local relative
    # In HTML file located at docs/Project_Report.html:
    # ../data/squad_eda_plots.png works, and screenshots/xxx.png works!
    
    # Pre-process LaTeX display equations $$ ... $$ into clean styled HTML containers
    def math_block_repl(match):
        eq = match.group(1).strip()
        return f'<div class="math-block">$$\\displaystyle {eq}$$</div>'
    md_text = re.sub(r'\$\$(.*?)\$\$', math_block_repl, md_text, flags=re.DOTALL)

    # Pre-process inline LaTeX $ ... $
    def inline_math_repl(match):
        eq = match.group(1).strip()
        return f'<span class="math-inline">${eq}$</span>'
    md_text = re.sub(r'(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)', inline_math_repl, md_text)

    # Convert Markdown to HTML with tables, fenced_code, and toc extensions
    html_body = markdown.markdown(
        md_text,
        extensions=['tables', 'fenced_code', 'attr_list', 'def_list', 'nl2br']
    )

    # Wrap figures nicely
    # Replace <p><img src="..." alt="..."><em>Figure ...</em></p> with clean figure containers
    def figure_wrapper(match):
        img_tag = match.group(1)
        caption = match.group(2)
        return f'<div class="figure-container">{img_tag}<p class="figure-caption">{caption}</p></div>'

    html_body = re.sub(
        r'<p>(<img[^>]+>)\s*<em>(Figure[^<]+)</em></p>',
        figure_wrapper,
        html_body,
        flags=re.IGNORECASE
    )

    # HTML Document Template
    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Project 12: Extractive Question Answering System - Academic Report</title>
    <!-- MathJax for rendering publication-grade KaTeX/LaTeX formulas -->
    <script>
    MathJax = {{
        tex: {{
            inlineMath: [['$', '$']],
            displayMath: [['$$', '$$']]
        }},
        svg: {{ fontCache: 'global' }}
    }};
    </script>
    <script type="text/javascript" id="MathJax-script" async
      src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js">
    </script>
    <style>
        @page {{
            size: A4 portrait;
            margin: 20mm 18mm 20mm 18mm;
            @bottom-right {{
                content: counter(page);
                font-family: 'Segoe UI', system-ui, sans-serif;
                font-size: 9pt;
                color: #64748B;
            }}
            @bottom-left {{
                content: "Project 12: Extractive Question Answering System (Group 10)";
                font-family: 'Segoe UI', system-ui, sans-serif;
                font-size: 9pt;
                color: #64748B;
            }}
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            font-size: 10.5pt;
            line-height: 1.6;
            color: #1E293B;
            background: #FFFFFF;
            margin: 0;
            padding: 0;
        }}

        h1 {{
            font-size: 22pt;
            font-weight: 800;
            color: #0F172A;
            line-height: 1.25;
            margin-top: 0;
            margin-bottom: 8px;
            text-align: center;
        }}

        .subtitle {{
            text-align: center;
            font-size: 13pt;
            color: #0B57D0;
            font-weight: 600;
            margin-bottom: 20px;
        }}

        .metadata-box {{
            background: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-left: 4px solid #0B57D0;
            border-radius: 8px;
            padding: 14px 18px;
            margin-bottom: 25px;
            font-size: 9.5pt;
        }}

        .metadata-box p {{
            margin: 4px 0;
        }}

        h2 {{
            font-size: 15pt;
            font-weight: 700;
            color: #0F172A;
            border-bottom: 1.5px solid #E2E8F0;
            padding-bottom: 6px;
            margin-top: 26px;
            margin-bottom: 12px;
            break-after: avoid;
        }}

        h3 {{
            font-size: 12pt;
            font-weight: 600;
            color: #1E3A8A;
            margin-top: 18px;
            margin-bottom: 8px;
            break-after: avoid;
        }}

        h4 {{
            font-size: 10.5pt;
            font-weight: 600;
            color: #334155;
            margin-top: 14px;
            margin-bottom: 6px;
            break-after: avoid;
        }}

        p, li {{
            text-align: justify;
            text-justify: inter-word;
        }}

        ul, ol {{
            margin-top: 6px;
            margin-bottom: 12px;
            padding-left: 24px;
        }}

        li {{
            margin-bottom: 4px;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 16px 0;
            font-size: 9.5pt;
            break-inside: avoid;
        }}

        th {{
            background: #0F172A;
            color: #FFFFFF;
            font-weight: 600;
            text-align: left;
            padding: 8px 12px;
            border: 1px solid #0F172A;
        }}

        td {{
            padding: 8px 12px;
            border: 1px solid #E2E8F0;
            background: #FFFFFF;
        }}

        tr:nth-child(even) td {{
            background: #F8FAFC;
        }}

        code {{
            font-family: 'Consolas', 'Courier New', monospace;
            font-size: 9pt;
            background: #F1F5F9;
            color: #0F172A;
            padding: 2px 5px;
            border-radius: 4px;
            border: 1px solid #E2E8F0;
        }}

        pre {{
            background: #0F172A;
            color: #F8FAFC;
            padding: 12px 16px;
            border-radius: 6px;
            font-size: 8.5pt;
            overflow-x: auto;
            break-inside: avoid;
        }}

        pre code {{
            background: transparent;
            color: #F8FAFC;
            border: none;
            padding: 0;
        }}

        .math-block {{
            background: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 6px;
            padding: 10px 14px;
            margin: 12px 0;
            text-align: center;
            break-inside: avoid;
            font-size: 10pt;
        }}

        .figure-container {{
            margin: 20px auto;
            text-align: center;
            break-inside: avoid;
        }}

        .figure-container img, img {{
            max-width: 96%;
            height: auto;
            border-radius: 8px;
            border: 1px solid #CBD5E1;
            box-shadow: 0 4px 10px rgba(15, 23, 42, 0.08);
            margin: 8px auto;
            display: block;
        }}

        .figure-caption {{
            font-size: 9pt;
            color: #475569;
            margin-top: 6px;
            margin-bottom: 16px;
            text-align: center;
            font-style: italic;
        }}

        blockquote {{
            border-left: 3.5px solid #0B57D0;
            margin: 14px 0;
            padding: 6px 16px;
            background: #F1F5F9;
            color: #334155;
            font-style: italic;
            border-radius: 0 6px 6px 0;
        }}

        hr {{
            border: none;
            border-top: 1px solid #E2E8F0;
            margin: 24px 0;
        }}

        .badge {{
            display: inline-block;
            padding: 2px 8px;
            font-size: 8pt;
            font-weight: 600;
            border-radius: 12px;
            background: #E0E7FF;
            color: #3730A3;
            margin-right: 4px;
        }}
    </style>
</head>
<body>
    {html_body}
</body>
</html>"""

    with open(REPORT_HTML, "w", encoding="utf-8") as f:
        f.write(html_doc)
    print(f"[+] Successfully compiled report HTML: {REPORT_HTML}")

    print(f"[*] Rendering publication PDF via Chrome headless...")
    cmd = [
        CHROME_PATH,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--run-all-compositor-stages-before-draw",
        f"--print-to-pdf={REPORT_PDF}",
        "--no-pdf-header-footer",
        REPORT_HTML
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    if os.path.exists(REPORT_PDF) and os.path.getsize(REPORT_PDF) > 0:
        size_kb = os.path.getsize(REPORT_PDF) / 1024
        print(f"[+] Publication PDF successfully generated at: {REPORT_PDF} ({size_kb:.1f} KB)")
    else:
        print(f"[-] Error generating PDF. Stderr: {res.stderr}")

if __name__ == "__main__":
    generate_pdf_report()
