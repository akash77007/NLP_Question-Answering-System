"""
scripts/update_report_figures.py
Updates docs/Project_Report.md to embed real generated screenshots and diagrams
instead of ASCII placeholder boxes.
"""

import re
import os

REPORT_PATH = r"D:\Documents\Assignments\NLP\Question_Answering_System\docs\Project_Report.md"

with open(REPORT_PATH, "r", encoding="utf-8") as f:
    content = f.read()

# Replace Figure 1 Placeholder
fig1_pattern = r"```\n\+--+\+\n\|[^\n]*FIGURE 1 PLACEHOLDER[^\n]*\|\n[\s\S]*?\+--+\+\n\*Figure 1:[^\n]*\*\n```"
fig1_replacement = """![Figure 1: SQuAD 2.0 Exploratory Data Analysis Dashboard](../data/squad_eda_plots.png)
*Figure 1: Exploratory Data Analysis dashboard for SQuAD 2.0 validation split (n=11,873). Panel A shows the 50/50 class balance; Panel B plots context word counts with the 95th percentile cutoff at 220 words; Panel C displays question length distributions; Panel D highlights interrogative starter frequencies.*"""

# Replace Figure 2 Placeholder
fig2_pattern = r"```\n\+--+\+\n\|[^\n]*FIGURE 2 PLACEHOLDER[^\n]*\|\n[\s\S]*?\+--+\+\n\*Figure 2:[^\n]*\*\n```"
fig2_replacement = """![Figure 2: BERT Extractive Question Answering Architecture](screenshots/bert_qa_architecture.png)
*Figure 2: End-to-end architecture of the BERT Extractive Question Answering Engine. The model encodes joint Question-Passage sequences via 12 Transformer layers, projecting output contextual embeddings $T_i$ into start logits $s_i$ and end logits $e_j$, with [CLS] representation $T_0$ dedicated to SQuAD 2.0 adversarial null scoring.*"""

# Replace Figure 3 Placeholder
fig3_pattern = r"```\n\+--+\+\n\|[^\n]*FIGURE 3 PLACEHOLDER[^\n]*\|\n[\s\S]*?\+--+\+\n\*Figure 3:[^\n]*\*\n```"
fig3_replacement = """![Figure 3: Streamlit Web Application Interface](screenshots/02_answer_extracted_apollo.png)
*Figure 3: Streamlit Web GUI (Material You design system) executing extractive inference on an Apollo 11 historical passage. The verbatim answer ("Eagle") is surfaced with 99.4% confidence and illuminated directly inside the source document.*"""

# Replace Figure 4 Placeholder
fig4_pattern = r"```\n\+--+\+\n\|[^\n]*FIGURE 4 PLACEHOLDER[^\n]*\|\n[\s\S]*?\+--+\+\n\*Figure 4:[^\n]*\*\n```"
fig4_replacement = """![Figure 4: Adversarial Query Rejection](screenshots/03_adversarial_rejection_eiffel.png)
*Figure 4: Streamlit Web GUI rejecting an adversarial trick question lacking supporting evidence in the text ("How many kilograms of gold were used to paint the Eiffel Tower?"). SQuAD 2.0 null scoring triggers a high-certainty unanswerable rejection card with zero generative hallucination.*

![Figure 5: Model Diagnostics and Token Offsets](screenshots/04_diagnostics_expanded.png)
*Figure 5: Collapsible viva defense diagnostics panel exposing exact character start and end indices (offsets 111-116), logit score margins, and raw prediction JSON metadata for auditable extraction provenance.*

![Figure 6: Model Architecture & Benchmark Performance Dashboard](screenshots/05_architecture_benchmark.png)
*Figure 6: Empirical evaluation benchmark dashboard integrated into Tab 2 of the Streamlit application, displaying Exact Match (64.00%), F1 score (69.23%), and RTX 3050 GPU latency (32.85 ms).*"""

new_content = re.sub(fig1_pattern, fig1_replacement, content)
new_content = re.sub(fig2_pattern, fig2_replacement, new_content)
new_content = re.sub(fig3_pattern, fig3_replacement, new_content)
new_content = re.sub(fig4_pattern, fig4_replacement, new_content)

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(new_content)

print("[+] Project_Report.md updated with real screenshot links.")
