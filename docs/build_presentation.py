"""
docs/build_presentation.py
Generates the official 10-slide PowerPoint presentation (.pptx)
matching all requirements from Page 2 of the Capstone Guidelines.
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor

def create_presentation():
    prs = Presentation()
    # 16:9 Widescreen dimensions
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    # Theme Colors
    NAVY = RGBColor(30, 58, 138)       # #1E3A8A
    DARK_GRAY = RGBColor(55, 65, 81)   # #374151
    EMERALD = RGBColor(16, 185, 129)   # #10B981
    LIGHT_BG = RGBColor(243, 244, 246) # #F3F4F6

    blank_layout = prs.slide_layouts[6]

    def add_header(slide, title_text, category_text="Project 12: Question Answering System"):
        # Header Box
        tx_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(1.1))
        tf = tx_box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

        p_cat = tf.paragraphs[0]
        p_cat.text = category_text.upper()
        p_cat.font.size = Pt(11)
        p_cat.font.bold = True
        p_cat.font.color.rgb = EMERALD

        p_title = tf.add_paragraph()
        p_title.text = title_text
        p_title.font.size = Pt(26)
        p_title.font.bold = True
        p_title.font.color.rgb = NAVY

    # -------------------------------------------------------------
    # Slide 1: Title Slide
    # -------------------------------------------------------------
    s1 = prs.slides.add_slide(blank_layout)
    t_box = s1.shapes.add_textbox(Inches(1.0), Inches(0.8), Inches(11.3), Inches(5.8))
    tf1 = t_box.text_frame
    tf1.word_wrap = True

    p = tf1.paragraphs[0]
    p.text = "NLP CAPSTONE PROJECT 12 — GROUP 10"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = EMERALD

    p2 = tf1.add_paragraph()
    p2.text = "Extractive Question Answering System"
    p2.font.size = Pt(32)
    p2.font.bold = True
    p2.font.color.rgb = NAVY

    p3 = tf1.add_paragraph()
    p3.text = "Context-Grounded Reading Comprehension with BERT and SQuAD 2.0"
    p3.font.size = Pt(18)
    p3.font.color.rgb = DARK_GRAY

    p4 = tf1.add_paragraph()
    p4.text = (
        "\nTeam Members (Group 10):\n"
        "• Roll No: 2401201113        • Roll No: 2401201213\n"
        "• Roll No: 2401201115        • Roll No: 2401201104\n"
        "• Roll No: 2401201102        • Roll No: 2401201108 (Akash Sharma)\n\n"
        "GitHub Repository: https://github.com/akash77007/NLP_Question-Answering-System\n"
        "Course: BCA (AI & Data Science) | Hardware: NVIDIA RTX 3050 Laptop GPU (CUDA)"
    )
    p4.font.size = Pt(13)
    p4.font.color.rgb = DARK_GRAY

    # -------------------------------------------------------------
    # Slide 2: Problem Statement & Objectives
    # -------------------------------------------------------------
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "1. Problem Statement & Key Objectives")
    b2 = s2.shapes.add_textbox(Inches(0.8), Inches(1.7), Inches(11.7), Inches(5.2))
    tf2 = b2.text_frame
    tf2.word_wrap = True

    points_s2 = [
        ("The Factual Hallucination Problem in Modern NLP:", True),
        ("Generative LLMs frequently hallucinate facts or bring in unverified web training data.", False),
        ("In high-stakes enterprise, legal, and clinical settings, answers must be strictly grounded in provided context.", False),
        ("Key Objectives of Project 12:", True),
        ("Build a purely extractive question answering system that extracts verbatim spans from source text.", False),
        ("Implement SQuAD 2.0 unanswerable question detection to reject trick queries with no factual evidence.", False),
        ("Eliminate hallucination by ensuring zero words outside the source document are returned.", False),
        ("Deliver a high-throughput, GPU-accelerated Streamlit web application with real-time in-passage highlighting.", False)
    ]
    for text, is_bold in points_s2:
        para = tf2.add_paragraph()
        para.text = f"•  {text}" if not is_bold else text
        para.font.size = Pt(15 if not is_bold else 17)
        para.font.bold = is_bold
        para.font.color.rgb = NAVY if is_bold else DARK_GRAY
        para.space_after = Pt(6)

    # -------------------------------------------------------------
    # Slide 3: Introduction & Real-World Applications
    # -------------------------------------------------------------
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "2. Introduction & Real-World Applications")
    b3 = s3.shapes.add_textbox(Inches(0.8), Inches(1.7), Inches(11.7), Inches(5.2))
    tf3 = b3.text_frame
    tf3.word_wrap = True

    points_s3 = [
        ("What is Extractive Question Answering?", True),
        ("A reading comprehension technique that predicts the exact start and end token boundaries of the answer within a passage.", False),
        ("Guarantees 100% auditability: every prediction can be verified against the source text.", False),
        ("Core Industry Applications:", True),
        ("Legal Contract Analysis: Locating liability limits, indemnification clauses, and expiration dates in contracts.", False),
        ("Healthcare & Clinical Support: Extracting drug dosages, contraindications, and treatment protocols from medical journals.", False),
        ("Financial Document Audit: Querying SEC 10-K filings for revenue figures, debt metrics, and risk factors.", False),
        ("Enterprise Knowledge Search: Powering internal documentation search with zero risk of fabricated information.", False)
    ]
    for text, is_bold in points_s3:
        para = tf3.add_paragraph()
        para.text = f"•  {text}" if not is_bold else text
        para.font.size = Pt(15 if not is_bold else 17)
        para.font.bold = is_bold
        para.font.color.rgb = NAVY if is_bold else DARK_GRAY
        para.space_after = Pt(6)

    # -------------------------------------------------------------
    # Slide 4: Dataset Description (Stanford SQuAD 2.0)
    # -------------------------------------------------------------
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "3. Dataset Description: Stanford SQuAD 2.0")
    b4 = s4.shapes.add_textbox(Inches(0.8), Inches(1.7), Inches(11.7), Inches(5.2))
    tf4 = b4.text_frame
    tf4.word_wrap = True

    points_s4 = [
        ("Stanford Question Answering Dataset 2.0 (SQuAD 2.0):", True),
        ("Training Set: 130,319 question-passage pairs from Wikipedia articles.", False),
        ("Validation / Dev Set: 11,873 question-passage pairs with multiple human reference answers.", False),
        ("Why SQuAD 2.0 instead of SQuAD 1.1?", True),
        ("SQuAD 1.1 assumed every question had an answer in the passage, forcing models to guess on irrelevant queries.", False),
        ("SQuAD 2.0 introduced over 50,000 adversarial, unanswerable questions written by human annotators.", False),
        ("These negative examples look plausible and share key entities with the text, but the text lacks the actual answer.", False),
        ("Teaches the model a crucial capability: knowing when to say 'No answer found in the passage'.", False)
    ]
    for text, is_bold in points_s4:
        para = tf4.add_paragraph()
        para.text = f"•  {text}" if not is_bold else text
        para.font.size = Pt(15 if not is_bold else 17)
        para.font.bold = is_bold
        para.font.color.rgb = NAVY if is_bold else DARK_GRAY
        para.space_after = Pt(6)

    # -------------------------------------------------------------
    # Slide 5: Data Preprocessing & EDA (with embedded image)
    # -------------------------------------------------------------
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "4. Data Preprocessing & Exploratory Data Analysis")
    b5 = s5.shapes.add_textbox(Inches(0.8), Inches(1.7), Inches(6.0), Inches(5.2))
    tf5 = b5.text_frame
    tf5.word_wrap = True

    points_s5 = [
        ("Key EDA Discoveries (SQuAD 2.0 Validation):", True),
        ("Class Balance: 49.93% Answerable vs. 50.07% Unanswerable (perfectly balanced 50/50 benchmark).", False),
        ("Passage Lengths: Median = 120 words; 95th Percentile = 220 words. Comfortably fits inside BERT's 384-token window.", False),
        ("Question Lengths: Median = 10 words; 95th Percentile = 15 words.", False),
        ("Question Types: 'What' (58%), 'Who' (14%), 'How' (10%), 'When' (9%), 'Where' (5%).", False),
        ("Preprocessing Pipeline:", True),
        ("WordPiece Subword Tokenizer (30,522 vocabulary size).", False),
        ("Pair sequence formatting: [CLS] Question [SEP] Context [SEP].", False),
        ("Preservation of character offset mappings for verbatim text retrieval.", False)
    ]
    for text, is_bold in points_s5:
        para = tf5.add_paragraph()
        para.text = f"•  {text}" if not is_bold else text
        para.font.size = Pt(13 if not is_bold else 15)
        para.font.bold = is_bold
        para.font.color.rgb = NAVY if is_bold else DARK_GRAY
        para.space_after = Pt(4)

    # Embed EDA Chart on right half
    eda_img = os.path.join("data", "squad_eda_plots.png")
    if os.path.exists(eda_img):
        s5.shapes.add_picture(eda_img, Inches(7.1), Inches(1.7), width=Inches(5.4))

    # -------------------------------------------------------------
    # Slide 6: Feature Engineering & Methodology
    # -------------------------------------------------------------
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "5. Feature Engineering & Transformer Methodology")
    b6 = s6.shapes.add_textbox(Inches(0.8), Inches(1.7), Inches(11.7), Inches(5.2))
    tf6 = b6.text_frame
    tf6.word_wrap = True

    points_s6 = [
        ("Bidirectional Contextual Representation:", True),
        ("Traditional RNNs/LSTMs read text left-to-right or right-to-left sequentially.", False),
        ("BERT uses Transformer multi-head self-attention to process both left and right context simultaneously.", False),
        ("Attention Formula: Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) * V", False),
        ("Triple Input Representation for Question Answering:", True),
        ("Token Embeddings: Subword WordPiece representations capturing morphological roots.", False),
        ("Segment Embeddings (Token Type IDs): Binary mask informing attention layers which tokens belong to the Question (0) and Passage (1).", False),
        ("Position Embeddings: Vector representations encoding absolute token position in the 384-token sequence.", False)
    ]
    for text, is_bold in points_s6:
        para = tf6.add_paragraph()
        para.text = f"•  {text}" if not is_bold else text
        para.font.size = Pt(15 if not is_bold else 17)
        para.font.bold = is_bold
        para.font.color.rgb = NAVY if is_bold else DARK_GRAY
        para.space_after = Pt(6)

    # -------------------------------------------------------------
    # Slide 7: Model Architecture & Algorithms
    # -------------------------------------------------------------
    s7 = prs.slides.add_slide(blank_layout)
    add_header(s7, "6. Model Architecture & Span Extraction Algorithms")
    b7 = s7.shapes.add_textbox(Inches(0.8), Inches(1.7), Inches(11.7), Inches(5.2))
    tf7 = b7.text_frame
    tf7.word_wrap = True

    points_s7 = [
        ("Model Backbone: BERT-base-cased (110M Parameters, 12 Layers, 768 Hidden, 12 Heads)", True),
        ("Span Classification Head:", True),
        ("A dense linear projection (R^768 -> R^2) computes two logits for every token position k:", False),
        ("Start Logit: s_k = w_start^T * h_k   |   End Logit: e_k = w_end^T * h_k", False),
        ("Span Scoring & Constraint Algorithm:", True),
        ("Finds token indices (i, j) that maximize: Span Score S_(i, j) = s_i + e_j", False),
        ("Constraints: i <= j, span length <= 30 tokens, and tokens must reside in Context (not Question/Special tokens).", False),
        ("Adversarial Unanswerable Detection Mechanism:", True),
        ("Null Score at [CLS] token: S_null = s_0 + e_0 (trained as target for unanswerable questions).", False),
        ("Decision Rule: If (max(S_(i, j)) - S_null) < Threshold -> Output 'No answer found in passage'.", False)
    ]
    for text, is_bold in points_s7:
        para = tf7.add_paragraph()
        para.text = f"•  {text}" if not is_bold else text
        para.font.size = Pt(14 if not is_bold else 16)
        para.font.bold = is_bold
        para.font.color.rgb = NAVY if is_bold else DARK_GRAY
        para.space_after = Pt(5)

    # -------------------------------------------------------------
    # Slide 8: Performance Evaluation & Results
    # -------------------------------------------------------------
    s8 = prs.slides.add_slide(blank_layout)
    add_header(s8, "7. Performance Evaluation & Benchmark Results")
    b8 = s8.shapes.add_textbox(Inches(0.8), Inches(1.7), Inches(11.7), Inches(5.2))
    tf8 = b8.text_frame
    tf8.word_wrap = True

    points_s8 = [
        ("Empirical Benchmark Results (Evaluated on SQuAD 2.0 Dev Split):", True),
        ("Overall Exact Match (EM): 64.00%", False),
        ("Overall Token-Level F1 Score: 69.23%", False),
        ("Unanswerable (No-Answer) Accuracy: 74.00%", False),
        ("Answerable (Has-Answer) F1 Score: 64.46%", False),
        ("Hardware Latency: 32.85 ms / query on NVIDIA GeForce RTX 3050 Laptop GPU (~30 QPS)", False),
        ("Evaluation Metric Definitions:", True),
        ("Exact Match (EM): Binary 1/0 metric. Prediction must match ground truth identically after lowercasing and article stripping.", False),
        ("Token F1: Harmonic mean of precision and recall over word tokens. Handles partial boundary overlaps gracefully.", False),
        ("Error Analysis: Most non-EM predictions were semantically identical with minor boundary variations (e.g., EU Court of Justice).", False)
    ]
    for text, is_bold in points_s8:
        para = tf8.add_paragraph()
        para.text = f"•  {text}" if not is_bold else text
        para.font.size = Pt(14 if not is_bold else 16)
        para.font.bold = is_bold
        para.font.color.rgb = NAVY if is_bold else DARK_GRAY
        para.space_after = Pt(5)

    # -------------------------------------------------------------
    # Slide 9: GUI Demonstration & Architecture
    # -------------------------------------------------------------
    s9 = prs.slides.add_slide(blank_layout)
    add_header(s9, "8. Web GUI Demonstration (Streamlit)")
    b9 = s9.shapes.add_textbox(Inches(0.8), Inches(1.7), Inches(11.7), Inches(5.2))
    tf9 = b9.text_frame
    tf9.word_wrap = True

    points_s9 = [
        ("Interactive Web Application Features (Streamlit):", True),
        ("Document Input Text Area: Supports pasting any custom context passage with live word and character counters.", False),
        ("Curated Preset Scenarios: One-click selection for Science, Space History, AI History, and Adversarial Trick questions.", False),
        ("In-Passage Visual Highlighter: Uses custom HTML span badges to visually illuminate the exact extracted answer in the context.", False),
        ("Metrics Dashboard: Displays Extraction Status, Calibrated Confidence %, Logit Margin, and Latency in milliseconds.", False),
        ("Dynamic Sensitivity Slider: Sidebar threshold control to demonstrate strict vs. lenient unanswerable behavior live.", False),
        ("GPU Memory Caching: Leverages @st.cache_resource to load BERT once into VRAM, enabling instant sub-40ms user interactions.", False)
    ]
    for text, is_bold in points_s9:
        para = tf9.add_paragraph()
        para.text = f"•  {text}" if not is_bold else text
        para.font.size = Pt(15 if not is_bold else 17)
        para.font.bold = is_bold
        para.font.color.rgb = NAVY if is_bold else DARK_GRAY
        para.space_after = Pt(6)

    # -------------------------------------------------------------
    # Slide 10: Conclusion & Future Scope
    # -------------------------------------------------------------
    s10 = prs.slides.add_slide(blank_layout)
    add_header(s10, "9. Conclusion & Future Scope")
    b10 = s10.shapes.add_textbox(Inches(0.8), Inches(1.7), Inches(11.7), Inches(5.2))
    tf10 = b10.text_frame
    tf10.word_wrap = True

    points_s10 = [
        ("Conclusion:", True),
        ("Successfully developed, evaluated, and deployed an extractive Question Answering system using BERT and SQuAD 2.0.", False),
        ("Achieved strict factual grounding with zero generative hallucination.", False),
        ("Demonstrated robust rejection of adversarial unanswerable questions (74% accuracy).", False),
        ("High-speed real-time performance (~32 ms latency) on local consumer GPU hardware.", False),
        ("Future Scope & Extensions:", True),
        ("Multi-Document Retrieval (RAG): Integrating dense vector indexing (FAISS) to scale from single passages to full document libraries.", False),
        ("Edge Quantization: Converting BERT to INT8 with ONNX Runtime to enable lightweight edge and mobile execution.", False),
        ("Multilingual QA: Adopting XLM-RoBERTa for cross-lingual reading comprehension across Indian vernacular languages.", False)
    ]
    for text, is_bold in points_s10:
        para = tf10.add_paragraph()
        para.text = f"•  {text}" if not is_bold else text
        para.font.size = Pt(15 if not is_bold else 17)
        para.font.bold = is_bold
        para.font.color.rgb = NAVY if is_bold else DARK_GRAY
        para.space_after = Pt(6)

    out_file = os.path.join("docs", "Project_12_Presentation.pptx")
    prs.save(out_file)
    print(f"[+] Successfully generated 10-slide presentation at: {out_file}")

if __name__ == "__main__":
    create_presentation()
