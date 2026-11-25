"""
docs/build_presentation.py
Professional PowerPoint Presentation Generator (16:9 Widescreen)
Project 12: Extractive Question Answering System (Group 10)
Uses polished executive cards, real native tables, embedded EDA plots,
and clear screenshot/diagram placeholders.
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.dml.color import RGBColor

def create_presentation():
    prs = Presentation()
    # 16:9 Widescreen format
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    blank_layout = prs.slide_layouts[6]

    # Color Palette (Executive Navy, Royal Blue, Emerald Green, Slate Gray)
    NAVY_DARK    = RGBColor(15, 23, 42)      # #0F172A
    PRIMARY_BLUE = RGBColor(11, 87, 208)    # #0B57D0
    ACCENT_TEAL  = RGBColor(16, 185, 129)   # #10B981
    DARK_GRAY    = RGBColor(51, 65, 85)     # #334155
    MUTED_GRAY   = RGBColor(100, 116, 139)  # #64748B
    CARD_BG      = RGBColor(248, 250, 252)  # #F8FAFC
    CARD_BORDER  = RGBColor(226, 232, 240)  # #E2E8F0
    WHITE        = RGBColor(255, 255, 255)
    RED_ACCENT   = RGBColor(220, 38, 38)    # #DC2626
    ROSE_BG      = RGBColor(254, 242, 242)  # #FEF2F2
    GREEN_BG     = RGBColor(240, 253, 244)  # #F0FDF4

    def add_slide_header(slide, title_text, category_text="PROJECT 12: QUESTION ANSWERING SYSTEM"):
        """Adds a standardized, polished executive header."""
        header_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.95))
        tf = header_box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

        p_cat = tf.paragraphs[0]
        p_cat.text = category_text.upper()
        p_cat.font.size = Pt(10)
        p_cat.font.bold = True
        p_cat.font.color.rgb = ACCENT_TEAL

        p_title = tf.add_paragraph()
        p_title.text = title_text
        p_title.font.size = Pt(24)
        p_title.font.bold = True
        p_title.font.color.rgb = NAVY_DARK

    def add_card(slide, left, top, width, height, fill_color=CARD_BG, border_color=CARD_BORDER):
        """Creates a modern rounded card surface."""
        shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        shape.fill.solid()
        shape.fill.fore_color.rgb = fill_color
        if border_color:
            shape.line.color.rgb = border_color
            shape.line.width = Pt(1.2)
        else:
            shape.line.fill.background()
        return shape

    def add_placeholder_box(slide, left, top, width, height, label_text):
        """Creates an explicit image/screenshot placeholder box."""
        box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        box.fill.solid()
        box.fill.fore_color.rgb = RGBColor(241, 245, 249) # Light slate
        box.line.color.rgb = RGBColor(148, 163, 184)     # Slate border
        box.line.width = Pt(1.5)

        tf = box.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE

        p = tf.paragraphs[0]
        p.text = "📷 SCREENSHOT / IMAGE PLACEHOLDER"
        p.alignment = PP_ALIGN.CENTER
        p.font.size = Pt(12)
        p.font.bold = True
        p.font.color.rgb = PRIMARY_BLUE

        p2 = tf.add_paragraph()
        p2.text = label_text
        p2.alignment = PP_ALIGN.CENTER
        p2.font.size = Pt(10)
        p2.font.color.rgb = MUTED_GRAY
        return box

    # =============================================================
    # SLIDE 1: Title Slide (Hero Presentation)
    # =============================================================
    s1 = prs.slides.add_slide(blank_layout)
    # Hero card surface
    hero = add_card(s1, Inches(0.8), Inches(0.8), Inches(11.733), Inches(5.9), fill_color=NAVY_DARK, border_color=None)

    t_box = s1.shapes.add_textbox(Inches(1.3), Inches(1.2), Inches(10.7), Inches(5.1))
    tf1 = t_box.text_frame
    tf1.word_wrap = True

    p = tf1.paragraphs[0]
    p.text = "NLP CAPSTONE PROJECT 12  •  GROUP 10"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = ACCENT_TEAL

    p2 = tf1.add_paragraph()
    p2.text = "DocuBERT: Context-Grounded\nExtractive Question Answering"
    p2.font.size = Pt(32)
    p2.font.bold = True
    p2.font.color.rgb = WHITE
    p2.space_after = Pt(8)

    p3 = tf1.add_paragraph()
    p3.text = "High-Precision Reading Comprehension with BERT-base and Stanford SQuAD 2.0"
    p3.font.size = Pt(16)
    p3.font.color.rgb = RGBColor(203, 213, 225) # Slate 300
    p3.space_after = Pt(16)

    p4 = tf1.add_paragraph()
    p4.text = (
        "Team Members (Group 10):\n"
        "• Akash Sharma (Project Lead): 2401201108   • Khushi: 2401201102\n"
        "• Aryan Sareen: 2401201104                  • Aman: 2401201115\n"
        "• Vivek Sharma: 2401201213\n\n"
        "GitHub: https://github.com/akash77007/NLP_Question-Answering-System\n"
        "Demo Video: https://drive.google.com/file/d/1c_FlTJh2mWxDgKP-5dGBlNSZ4vWIuDO9/view?usp=sharing\n"
        "Course: BCA (AI & Data Science) | Hardware: NVIDIA RTX 3050 Laptop GPU (CUDA 13.x)"
    )
    p4.font.size = Pt(12)
    p4.font.color.rgb = RGBColor(148, 163, 184) # Slate 400

    # =============================================================
    # SLIDE 2: Problem Statement & Objectives
    # =============================================================
    s2 = prs.slides.add_slide(blank_layout)
    add_slide_header(s2, "1. Problem Statement & Key Objectives")

    # Left Card: The Problem
    add_card(s2, Inches(0.8), Inches(1.55), Inches(5.6), Inches(5.3), fill_color=ROSE_BG, border_color=RGBColor(254, 202, 202))
    b2_left = s2.shapes.add_textbox(Inches(1.0), Inches(1.75), Inches(5.2), Inches(4.8))
    tf2_l = b2_left.text_frame
    tf2_l.word_wrap = True

    p = tf2_l.paragraphs[0]
    p.text = "THE PROBLEM: GENERATIVE HALLUCINATION"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = RED_ACCENT
    p.space_after = Pt(10)

    problem_points = [
        "Factual Inaccuracy: Generative LLMs synthesize fluent text but frequently hallucinate non-existent facts.",
        "Missing Provenance: Answers lack verbatim traceability to source documents, making audits impossible.",
        "False Positive Bias: Traditional QA models blindly guess answers even when reference text lacks evidence.",
        "High Enterprise Risk: In legal contracts, clinical medicine, and financial filings, hallucinated text causes liability."
    ]
    for pt in problem_points:
        para = tf2_l.add_paragraph()
        para.text = f"•  {pt}"
        para.font.size = Pt(13)
        para.font.color.rgb = DARK_GRAY
        para.space_after = Pt(8)

    # Right Card: Project Objectives
    add_card(s2, Inches(6.8), Inches(1.55), Inches(5.7), Inches(5.3), fill_color=GREEN_BG, border_color=RGBColor(187, 247, 208))
    b2_right = s2.shapes.add_textbox(Inches(7.0), Inches(1.75), Inches(5.3), Inches(4.8))
    tf2_r = b2_right.text_frame
    tf2_r.word_wrap = True

    p = tf2_r.paragraphs[0]
    p.text = "PROJECT 12 OBJECTIVES & SOLUTION"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = RGBColor(22, 101, 52)
    p.space_after = Pt(10)

    solution_points = [
        "Pure Extractive QA: Restrict answer predictions strictly to verbatim substrings physically present in the source passage.",
        "Zero Hallucination: Guarantee that no words outside the reference passage are ever returned.",
        "SQuAD 2.0 Adversarial Rejection: Reject questions lacking evidence using [CLS] null-thresholding.",
        "Low-Latency GPU Deployment: Deliver a responsive Streamlit web app operating at ~32 ms per query on NVIDIA RTX 3050."
    ]
    for pt in solution_points:
        para = tf2_r.add_paragraph()
        para.text = f"✔  {pt}"
        para.font.size = Pt(13)
        para.font.color.rgb = DARK_GRAY
        para.space_after = Pt(8)

    # =============================================================
    # SLIDE 3: Introduction & Real-World Applications
    # =============================================================
    s3 = prs.slides.add_slide(blank_layout)
    add_slide_header(s3, "2. Introduction & Real-World Applications")

    # 4 Grid Cards
    apps = [
        ("⚖️ Legal Contract Review", 
         "Locating indemnification terms, liability caps, and termination clauses across 100+ page master service agreements with verbatim auditability."),
        ("🏥 Clinical Healthcare Support", 
         "Retrieving exact pharmaceutical dosages, contraindications, and clinical trial results from medical literature without risking fabricated guidance."),
        ("📈 Financial Document Audit", 
         "Extracting balance sheet figures, EBITDA metrics, debt covenants, and risk disclosures directly from corporate SEC 10-K filings."),
        ("🏢 Enterprise Knowledge Search", 
         "Powering internal employee and customer support knowledge bases where answers must link directly to official documentation.")
    ]
    
    positions = [
        (Inches(0.8), Inches(1.6), Inches(5.6), Inches(2.4)),
        (Inches(6.8), Inches(1.6), Inches(5.7), Inches(2.4)),
        (Inches(0.8), Inches(4.3), Inches(5.6), Inches(2.5)),
        (Inches(6.8), Inches(4.3), Inches(5.7), Inches(2.5)),
    ]

    for (title, desc), (l, t, w, h) in zip(apps, positions):
        add_card(s3, l, t, w, h, fill_color=CARD_BG)
        box = s3.shapes.add_textbox(l + Inches(0.2), t + Inches(0.2), w - Inches(0.4), h - Inches(0.4))
        tf = box.text_frame
        tf.word_wrap = True
        p_t = tf.paragraphs[0]
        p_t.text = title
        p_t.font.size = Pt(15)
        p_t.font.bold = True
        p_t.font.color.rgb = PRIMARY_BLUE
        p_t.space_after = Pt(6)

        p_d = tf.add_paragraph()
        p_d.text = desc
        p_d.font.size = Pt(12)
        p_d.font.color.rgb = DARK_GRAY

    # =============================================================
    # SLIDE 4: Dataset Description (Stanford SQuAD 2.0)
    # =============================================================
    s4 = prs.slides.add_slide(blank_layout)
    add_slide_header(s4, "3. Dataset Description: Stanford SQuAD 2.0")

    # 3 Stat Cards on top
    stat_data = [
        ("130,319", "TRAINING EXAMPLES", "Extracted from Wikipedia articles across hundreds of diverse topics."),
        ("11,873", "VALIDATION SAMPLES", "Development split with multiple independent human reference annotations."),
        ("50,000+", "ADVERSARIAL QUESTIONS", "Unanswerable questions crafted by annotators to look superficially plausible.")
    ]
    stat_lefts = [Inches(0.8), Inches(4.8), Inches(8.8)]
    for (num, title, sub), l in zip(stat_data, stat_lefts):
        add_card(s4, l, Inches(1.55), Inches(3.7), Inches(1.6), fill_color=RGBColor(241, 245, 249))
        box = s4.shapes.add_textbox(l + Inches(0.15), Inches(1.65), Inches(3.4), Inches(1.4))
        tf = box.text_frame
        tf.word_wrap = True
        p_num = tf.paragraphs[0]
        p_num.text = num
        p_num.font.size = Pt(22)
        p_num.font.bold = True
        p_num.font.color.rgb = PRIMARY_BLUE
        
        p_t = tf.add_paragraph()
        p_t.text = title
        p_t.font.size = Pt(10)
        p_t.font.bold = True
        p_t.font.color.rgb = DARK_GRAY

        p_s = tf.add_paragraph()
        p_s.text = sub
        p_s.font.size = Pt(9.5)
        p_s.font.color.rgb = MUTED_GRAY

    # Comparison card below
    add_card(s4, Inches(0.8), Inches(3.4), Inches(11.7), Inches(3.4), fill_color=CARD_BG)
    b4_bot = s4.shapes.add_textbox(Inches(1.0), Inches(3.55), Inches(11.3), Inches(3.1))
    tf4_b = b4_bot.text_frame
    tf4_b.word_wrap = True

    p = tf4_b.paragraphs[0]
    p.text = "WHY SQUAD 2.0 INSTEAD OF SQUAD 1.1?"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = NAVY_DARK
    p.space_after = Pt(8)

    sq_points = [
        ("SQuAD 1.1 Flaw: Every question had an answer in the passage. Models simply learned surface lexical overlap and blindly guessed spans when given nonsense.", False),
        ("SQuAD 2.0 Solution: Combined 100k answerable questions with 50k+ adversarial unanswerable questions that cannot be answered using the passage.", False),
        ("Dataset Schema: Each entry contains 'context', 'question', 'answers' (list of strings), 'answer_start' (character offset), and 'is_impossible' (Boolean flag).", False),
        ("Core Capability Taught: The neural network is forced to confirm whether supporting evidence exists before extracting a span, preventing false positives.", False)
    ]
    for pt, bold in sq_points:
        para = tf4_b.add_paragraph()
        para.text = f"•  {pt}"
        para.font.size = Pt(12)
        para.font.color.rgb = DARK_GRAY
        para.space_after = Pt(6)

    # =============================================================
    # SLIDE 5: Data Preprocessing & EDA (with embedded chart)
    # =============================================================
    s5 = prs.slides.add_slide(blank_layout)
    add_slide_header(s5, "4. Data Preprocessing & Exploratory Data Analysis")

    # Left content box
    add_card(s5, Inches(0.8), Inches(1.55), Inches(6.0), Inches(5.3), fill_color=CARD_BG)
    b5_left = s5.shapes.add_textbox(Inches(1.0), Inches(1.7), Inches(5.6), Inches(5.0))
    tf5 = b5_left.text_frame
    tf5.word_wrap = True

    p = tf5.paragraphs[0]
    p.text = "KEY EDA TAKEAWAYS & PREPROCESSING"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = NAVY_DARK
    p.space_after = Pt(8)

    eda_points = [
        "Balanced Class Split: 49.93% Answerable vs. 50.07% Unanswerable (perfectly balanced 50/50 benchmark split).",
        "Passage Lengths: Median = 120 words; 95th Percentile = 220 words. Proves BERT's max_seq_length = 384 covers 95%+ of contexts without truncation.",
        "Question Lengths: Median = 10 words; 95th Percentile = 15 words.",
        "Question Types: 'What' (58.2%), 'Who' (14.1%), 'How' (10.3%), 'When' (9.1%), 'Where' (5.4%).",
        "WordPiece Tokenization: Subword packaging: [CLS] Question [SEP] Context [SEP].",
        "Lossless Offset Mapping: Stores (start_char, end_char) boundaries to preserve raw casing and whitespace.",
        "Zero Aggressive Cleaning: Stopwords and punctuation are preserved because BERT self-attention requires grammatical syntax."
    ]
    for pt in eda_points:
        para = tf5.add_paragraph()
        para.text = f"•  {pt}"
        para.font.size = Pt(11)
        para.font.color.rgb = DARK_GRAY
        para.space_after = Pt(5)

    # Right side: Embed the generated chart or show placeholder
    eda_img = os.path.join("data", "squad_eda_plots.png")
    if os.path.exists(eda_img):
        s5.shapes.add_picture(eda_img, Inches(7.1), Inches(1.55), width=Inches(5.4))
    else:
        add_placeholder_box(s5, Inches(7.1), Inches(1.55), Inches(5.4), Inches(5.3), "Paste SQuAD 2.0 4-Panel EDA Dashboard Here\n(File: data/squad_eda_plots.png)")

    # =============================================================
    # SLIDE 6: Feature Engineering & Transformer Methodology
    # =============================================================
    s6 = prs.slides.add_slide(blank_layout)
    add_slide_header(s6, "5. Feature Engineering & Methodology")

    # Left: Bidirectional Self-Attention
    add_card(s6, Inches(0.8), Inches(1.55), Inches(5.6), Inches(5.3), fill_color=CARD_BG)
    b6_l = s6.shapes.add_textbox(Inches(1.0), Inches(1.75), Inches(5.2), Inches(4.8))
    tf6_l = b6_l.text_frame
    tf6_l.word_wrap = True

    p = tf6_l.paragraphs[0]
    p.text = "BIDIRECTIONAL MULTI-HEAD ATTENTION"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = PRIMARY_BLUE
    p.space_after = Pt(8)

    attn_points = [
        "Eliminating Recurrence Bottlenecks: Unlike sequential LSTMs, BERT attends to full left and right context simultaneously across all 12 layers.",
        "Scaled Dot-Product Attention Formula:\nAttention(Q, K, V) = softmax((Q · Kᵀ) / √d_k) · V",
        "Multi-Head Mechanism: 12 independent attention heads allow the model to jointly attend to information from different representation subspaces at different positions.",
        "Cross-Attention Dynamics: Question tokens directly attend to candidate passage tokens, identifying syntactic and coreference links."
    ]
    for pt in attn_points:
        para = tf6_l.add_paragraph()
        para.text = f"•  {pt}"
        para.font.size = Pt(12)
        para.font.color.rgb = DARK_GRAY
        para.space_after = Pt(8)

    # Right: Triple Input Embeddings
    add_card(s6, Inches(6.8), Inches(1.55), Inches(5.7), Inches(5.3), fill_color=CARD_BG)
    b6_r = s6.shapes.add_textbox(Inches(7.0), Inches(1.75), Inches(5.3), Inches(4.8))
    tf6_r = b6_r.text_frame
    tf6_r.word_wrap = True

    p = tf6_r.paragraphs[0]
    p.text = "TRIPLE INPUT EMBEDDING REPRESENTATION"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = PRIMARY_BLUE
    p.space_after = Pt(8)

    emb_points = [
        "Summation Formulation:\nInput = Token_Emb + Segment_Emb + Position_Emb",
        "1. Token Embeddings: WordPiece subwords representing morphological roots across a 30,522 vocabulary size.",
        "2. Segment Embeddings (Token Type IDs): Binary mask assigning 0 to Question tokens and 1 to Context tokens, allowing the attention head to distinguish query from passage.",
        "3. Positional Embeddings: Learned 768-dimensional vectors encoding absolute token sequence position (0 to 383)."
    ]
    for pt in emb_points:
        para = tf6_r.add_paragraph()
        para.text = f"•  {pt}"
        para.font.size = Pt(12)
        para.font.color.rgb = DARK_GRAY
        para.space_after = Pt(8)

    # =============================================================
    # SLIDE 7: Model Architecture & Algorithms (With Diagram Placeholder)
    # =============================================================
    s7 = prs.slides.add_slide(blank_layout)
    add_slide_header(s7, "6. Model Architecture & Extraction Algorithms")

    # Left: Mathematical Span Formulation
    add_card(s7, Inches(0.8), Inches(1.55), Inches(6.4), Inches(5.3), fill_color=CARD_BG)
    b7_left = s7.shapes.add_textbox(Inches(1.0), Inches(1.7), Inches(6.0), Inches(5.0))
    tf7 = b7_left.text_frame
    tf7.word_wrap = True

    p = tf7.paragraphs[0]
    p.text = "SPAN CLASSIFICATION & NULL THRESHOLDING"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = NAVY_DARK
    p.space_after = Pt(6)

    arch_points = [
        "Model Backbone: 'deepset/bert-base-cased-squad2' (110M params, 12 layers, 768 hidden, 12 heads).",
        "QA Classification Head: Dense linear layer (R^768 -> R^2) applied to each token hidden state h_k:\n  • Start Logit: s_k = w_startᵀ · h_k\n  • End Logit: e_k = w_endᵀ · h_k",
        "Span Search Algorithm: Finds token indices (i, j) that maximize S_(i, j) = s_i + e_j subject to i <= j and tokens ∈ Context.",
        "Adversarial Null-Score at [CLS]: In SQuAD 2.0, impossible queries point to [CLS] (index 0). Null Score S_null = s_0 + e_0.",
        "Decision Rule: If (max(S_(i, j)) - S_null) < Threshold (tau = -1.0):\n  -> Output: 'No answer found in the provided passage.'",
        "Calibrated Confidence: Computed via Sigmoid(max(S_(i, j)) - S_null)."
    ]
    for pt in arch_points:
        para = tf7.add_paragraph()
        para.text = f"•  {pt}"
        para.font.size = Pt(11)
        para.font.color.rgb = DARK_GRAY
        para.space_after = Pt(5)

    # Right: Architecture Diagram
    arch_img = os.path.join("docs", "screenshots", "bert_qa_architecture.png")
    if os.path.exists(arch_img):
        s7.shapes.add_picture(arch_img, Inches(7.3), Inches(1.85), width=Inches(5.2))
    else:
        add_placeholder_box(s7, Inches(7.5), Inches(1.55), Inches(5.0), Inches(5.3), 
                            "Paste BERT Model Architecture Diagram Here\n([CLS] + Question + [SEP] + Passage ➔ BERT ➔ Logits)")

    # =============================================================
    # SLIDE 8: Performance Evaluation & Results (Native Styled Table)
    # =============================================================
    s8 = prs.slides.add_slide(blank_layout)
    add_slide_header(s8, "7. Performance Evaluation & Benchmark Results")

    # Native PowerPoint Table
    rows, cols = 6, 3
    left, top, width, height = Inches(0.8), Inches(1.55), Inches(11.733), Inches(2.6)
    table_shape = s8.shapes.add_table(rows, cols, left, top, width, height)
    table = table_shape.table

    # Column widths
    table.columns[0].width = Inches(5.5)
    table.columns[1].width = Inches(3.1)
    table.columns[2].width = Inches(3.133)

    # Table Data
    table_data = [
        ["Evaluation Metric Category", "Sample Test Size", "Empirical Score (%)"],
        ["Overall Exact Match (EM)", "100 Samples (Balanced)", "64.00%"],
        ["Overall Token-Level F1 Score", "100 Samples (Balanced)", "69.23%"],
        ["Unanswerable (No-Answer) Accuracy", "50 Impossible Samples", "74.00%"],
        ["Answerable (Has-Answer) F1 Score", "50 Answerable Samples", "64.46%"],
        ["Average Inference Latency", "NVIDIA RTX 3050 GPU", "32.85 ms / query (~30 QPS)"]
    ]

    for r_idx, row in enumerate(table_data):
        for c_idx, val in enumerate(row):
            cell = table.cell(r_idx, c_idx)
            cell.text = val
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = cell.text_frame.paragraphs[0]
            p.font.size = Pt(12)
            if r_idx == 0:
                p.font.bold = True
                p.font.color.rgb = WHITE
                cell.fill.solid()
                cell.fill.fore_color.rgb = PRIMARY_BLUE
            else:
                p.font.color.rgb = NAVY_DARK
                cell.fill.solid()
                cell.fill.fore_color.rgb = RGBColor(248, 250, 252) if r_idx % 2 == 1 else WHITE
                if c_idx == 2:
                    p.font.bold = True
                    p.font.color.rgb = PRIMARY_BLUE

    # Takeaways box below table
    add_card(s8, Inches(0.8), Inches(4.35), Inches(11.733), Inches(2.5), fill_color=CARD_BG)
    b8_bot = s8.shapes.add_textbox(Inches(1.0), Inches(4.5), Inches(11.3), Inches(2.2))
    tf8 = b8_bot.text_frame
    tf8.word_wrap = True

    p = tf8.paragraphs[0]
    p.text = "ERROR ANALYSIS & METRIC TAKEAWAYS"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = NAVY_DARK
    p.space_after = Pt(6)

    eval_takeaways = [
        "Exact Match vs. F1 Gap: Many answerable errors achieved EM = 0% but F1 > 80% due to minor boundary mismatches (e.g., extracting 'Court of Justice of the European Union' while human ground truth included '(CJEU)').",
        "Robust Unanswerable Precision: With threshold tau = -1.0, the model accurately rejected 74.00% of adversarial unanswerable questions.",
        "High GPU Throughput: Operating at 32.85 ms/query allows processing ~1,800 document queries per minute locally on a 4 GB VRAM GPU."
    ]
    for pt in eval_takeaways:
        para = tf8.add_paragraph()
        para.text = f"•  {pt}"
        para.font.size = Pt(11.5)
        para.font.color.rgb = DARK_GRAY
        para.space_after = Pt(4)

    # =============================================================
    # SLIDE 9: Web GUI Demonstration (With Screenshot Placeholder)
    # =============================================================
    s9 = prs.slides.add_slide(blank_layout)
    add_slide_header(s9, "8. Web GUI Demonstration (Streamlit & Material You)")

    # Left: Feature summary
    add_card(s9, Inches(0.8), Inches(1.55), Inches(5.6), Inches(5.3), fill_color=CARD_BG)
    b9_left = s9.shapes.add_textbox(Inches(1.0), Inches(1.7), Inches(5.2), Inches(5.0))
    tf9 = b9_left.text_frame
    tf9.word_wrap = True

    p = tf9.paragraphs[0]
    p.text = "STREAMLIT GUI CAPABILITIES"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = PRIMARY_BLUE
    p.space_after = Pt(6)

    gui_features = [
        "Document Text Area: Paste any custom article or report with live word and character counters.",
        "Curated Presets: One-click testing for Apollo 11, Photosynthesis, Turing AI, Legal Contracts, and Adversarial Trick questions.",
        "Visual In-Passage Highlighting: HTML span badges illuminate the exact extracted answer directly in the source document.",
        "Live Metrics Dashboard: Real-time cards for Extraction Status, Confidence %, Logit Margin, and Latency.",
        "Sensitivity Slider: Sidebar slider to adjust null threshold (tau) dynamically during live faculty defense.",
        "Resource Caching: Uses @st.cache_resource to keep BERT loaded in GPU VRAM, enabling sub-35ms responses.",
        "Live Demo Video: https://drive.google.com/file/d/1c_FlTJh2mWxDgKP-5dGBlNSZ4vWIuDO9/view?usp=sharing"
    ]
    for pt in gui_features:
        para = tf9.add_paragraph()
        para.text = f"•  {pt}"
        para.font.size = Pt(11)
        para.font.color.rgb = DARK_GRAY
        para.space_after = Pt(5)

    # Right: Streamlit Screenshot
    gui_img = os.path.join("docs", "screenshots", "02_answer_extracted_apollo.png")
    if os.path.exists(gui_img):
        s9.shapes.add_picture(gui_img, Inches(6.8), Inches(1.85), width=Inches(5.7))
    else:
        add_placeholder_box(s9, Inches(6.8), Inches(1.55), Inches(5.7), Inches(5.3),
                            "Paste Streamlit Web GUI Demo Screenshot Here\n(Showing passage input, question, highlighted answer, and confidence metrics)")

    # =============================================================
    # SLIDE 10: Conclusion & Future Scope
    # =============================================================
    s10 = prs.slides.add_slide(blank_layout)
    add_slide_header(s10, "9. Conclusion & Future Scope")

    # Left: Achievements
    add_card(s10, Inches(0.8), Inches(1.55), Inches(5.6), Inches(5.3), fill_color=GREEN_BG, border_color=RGBColor(187, 247, 208))
    b10_l = s10.shapes.add_textbox(Inches(1.0), Inches(1.75), Inches(5.2), Inches(4.8))
    tf10_l = b10_l.text_frame
    tf10_l.word_wrap = True

    p = tf10_l.paragraphs[0]
    p.text = "KEY PROJECT ACHIEVEMENTS"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = RGBColor(22, 101, 52)
    p.space_after = Pt(8)

    achieve_points = [
        "100% Extractive Grounding: Zero generative hallucination; every answer is an auditable substring from the user's text.",
        "Adversarial Rejection: SQuAD 2.0 null scoring reliably filters trick questions with missing evidence (74% accuracy).",
        "High-Speed Edge Execution: Sub-35ms inference on local RTX 3050 GPU hardware.",
        "Complete University Deliverables: Working GUI, frozen requirements, unit test suite, and empirical evaluation benchmarks."
    ]
    for pt in achieve_points:
        para = tf10_l.add_paragraph()
        para.text = f"✔  {pt}"
        para.font.size = Pt(12)
        para.font.color.rgb = DARK_GRAY
        para.space_after = Pt(8)

    # Right: Future Scope
    add_card(s10, Inches(6.8), Inches(1.55), Inches(5.7), Inches(5.3), fill_color=CARD_BG)
    b10_r = s10.shapes.add_textbox(Inches(7.0), Inches(1.75), Inches(5.3), Inches(4.8))
    tf10_r = b10_r.text_frame
    tf10_r.word_wrap = True

    p = tf10_r.paragraphs[0]
    p.text = "FUTURE ENHANCEMENTS"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = PRIMARY_BLUE
    p.space_after = Pt(8)

    future_points = [
        "Multi-Document Retrieval (RAG): Integrating dense vector indexing (FAISS with BGE embeddings) to scale from single passages to multi-document enterprise libraries.",
        "Model Quantization: Compiling BERT to INT8 with ONNX Runtime to reduce VRAM footprint below 200 MB for edge/mobile devices.",
        "Multilingual QA: Adopting XLM-RoBERTa for cross-lingual extractive reading comprehension across Indian vernacular languages (Hindi, Bengali, Tamil).",
        "Context Window Expansion: Implementing sliding-window stride chunking for 100+ page long documents."
    ]
    for pt in future_points:
        para = tf10_r.add_paragraph()
        para.text = f"🚀  {pt}"
        para.font.size = Pt(12)
        para.font.color.rgb = DARK_GRAY
        para.space_after = Pt(8)

    out_file = os.path.join("docs", "Project_12_Presentation.pptx")
    prs.save(out_file)
    print(f"[+] Successfully generated refined 10-slide presentation at: {out_file}")

if __name__ == "__main__":
    create_presentation()
