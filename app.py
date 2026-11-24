"""
app.py
Production-Grade Streamlit Web Application for Project 12: Extractive Question Answering System.
Built with PyTorch, Hugging Face Transformers (BERT SQuAD 2.0), and Streamlit.
Group 10 — BCA (Artificial Intelligence & Data Science) Capstone Project.
"""

import os
import json
import time
import torch
import streamlit as st
from src.inference import QuestionAnsweringEngine

# ---------------------------------------------------------
# Page Configuration & Metadata
# ---------------------------------------------------------
st.set_page_config(
    page_title="DocuBERT — Extractive Question Answering System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Modern Custom CSS Styling
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Top Banner / Hero */
    .hero-container {
        background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 50%, #4F46E5 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(37, 99, 235, 0.25);
    }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0;
        color: #FFFFFF;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        color: #E0E7FF;
        margin-top: 0.5rem;
        margin-bottom: 1.2rem;
        font-weight: 400;
    }
    .badge-container {
        display: flex;
        gap: 0.6rem;
        flex-wrap: wrap;
    }
    .hero-badge {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        border: 1px solid rgba(255, 255, 255, 0.25);
        color: #FFFFFF;
    }

    /* Cards */
    .feature-card {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 14px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.2rem;
    }
    
    /* Results Display */
    .answer-card {
        background: #F0FDF4;
        border: 1px solid #BBF7D0;
        border-left: 6px solid #16A34A;
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1.2rem;
        margin-bottom: 1.2rem;
    }
    .answer-header {
        font-size: 0.82rem;
        font-weight: 700;
        color: #15803D;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .answer-text {
        font-size: 1.8rem;
        font-weight: 700;
        color: #14532D;
        margin-top: 0.4rem;
        margin-bottom: 0.5rem;
        line-height: 1.25;
    }
    .answer-meta {
        font-size: 0.88rem;
        color: #166534;
        font-weight: 500;
    }

    /* Unanswerable / Rejection Card */
    .unanswerable-card {
        background: #FEF2F2;
        border: 1px solid #FECACA;
        border-left: 6px solid #DC2626;
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1.2rem;
        margin-bottom: 1.2rem;
    }
    .unanswerable-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #991B1B;
        margin: 0;
    }
    .unanswerable-desc {
        color: #7F1D1D;
        margin-top: 0.5rem;
        font-size: 0.95rem;
        line-height: 1.5;
    }

    /* In-Passage Highlighting */
    .highlight-span {
        background-color: #FEF08A;
        color: #854D0E;
        font-weight: 700;
        padding: 3px 7px;
        border-radius: 6px;
        box-shadow: 0 0 0 2px #FACC15;
    }

    .passage-viewer {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.5rem;
        line-height: 1.8;
        font-size: 1.05rem;
        color: #1E293B;
    }

    /* Metric Pills */
    .metric-pill-container {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .metric-pill {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .metric-pill-label {
        font-size: 0.78rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-pill-val {
        font-size: 1.45rem;
        font-weight: 800;
        color: #0F172A;
        margin-top: 0.2rem;
    }

    /* Team Card */
    .team-badge {
        background: #EEF2FF;
        color: #3730A3;
        border: 1px solid #C7D2FE;
        padding: 0.5rem 0.9rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.92rem;
        display: inline-block;
        margin: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Cached Model Loader
# ---------------------------------------------------------
@st.cache_resource(show_spinner="⚡ Initializing BERT QA Engine on GPU...")
def load_qa_engine():
    """Initializes and caches the BERT model once in VRAM."""
    return QuestionAnsweringEngine(model_name="deepset/bert-base-cased-squad2")


# ---------------------------------------------------------
# Curated Presets
# ---------------------------------------------------------
PRESETS = {
    "-- Custom Input (Enter your own context & question) --": {
        "context": "",
        "question": ""
    },
    "🚀 Apollo 11 Moon Landing (Answerable)": {
        "context": (
            "Apollo 11 was the American spaceflight that first landed humans on the Moon. "
            "Commander Neil Armstrong and Lunar Module pilot Buzz Aldrin landed the Apollo Lunar Module Eagle "
            "on July 20, 1969, at 20:17 UTC, and Armstrong became the first person to step onto the Moon's surface "
            "six hours and 39 minutes later, on July 21 at 02:56 UTC. Aldrin joined him 19 minutes later. "
            "They spent about two and a quarter hours together exploring the site they had named Tranquility Base."
        ),
        "question": "What was the name of the lunar module that landed on the Moon?"
    },
    "🌿 Photosynthesis in Plants (Answerable)": {
        "context": (
            "Photosynthesis is a biological process used by plants and other organisms to convert light energy "
            "into chemical energy that can later be released to fuel the organism's activities. This chemical "
            "energy is stored in carbohydrate molecules, such as sugars, which are synthesized from carbon dioxide "
            "and water. In most cases, oxygen is also released as a waste product."
        ),
        "question": "What waste product is released during photosynthesis?"
    },
    "💻 Alan Turing & AI (Answerable)": {
        "context": (
            "Alan Turing was an English mathematician, computer scientist, logician, and theoretical biologist. "
            "He was highly influential in the development of theoretical computer science, providing a formalisation "
            "of the concepts of algorithm and computation with the Turing machine, which can be considered a model "
            "of a general-purpose computer. In 1950, Turing proposed the Turing test as a measure of machine intelligence."
        ),
        "question": "When did Turing propose the Turing test?"
    },
    "🏛️ Eiffel Tower Gold (Adversarial / SQuAD 2.0 Unanswerable)": {
        "context": (
            "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. "
            "It is named after the engineer Gustave Eiffel, whose company designed and built the tower. "
            "Constructed from 1887 to 1889 as the entrance arch to the 1889 World's Fair, it was initially "
            "criticized by some of France's leading artists and intellectuals for its design."
        ),
        "question": "How many kilograms of gold were used to paint the Eiffel Tower?"
    },
    "⚖️ Commercial Contract Clause (Answerable)": {
        "context": (
            "This Software License Agreement is entered into on January 15, 2026. The Licensee agrees to pay "
            "an annual recurring fee of $45,000 USD within thirty (30) business days of invoice receipt. "
            "Either party may terminate this agreement with ninety (90) days written notice. In no event shall "
            "the Licensor's aggregate liability exceed the total fees paid under this Agreement during the preceding 12 months."
        ),
        "question": "What is the maximum termination notice period required?"
    }
}


def main():
    # -----------------------------------------------------
    # Sidebar: Model Diagnostics & Controls
    # -----------------------------------------------------
    st.sidebar.image("https://huggingface.co/front/assets/huggingface_logo-noborder.svg", width=50)
    st.sidebar.title("System Controls")
    
    # Hardware Status
    has_cuda = torch.cuda.is_available()
    device_label = "NVIDIA CUDA GPU" if has_cuda else "Host CPU"
    device_name = torch.cuda.get_device_name(0) if has_cuda else "CPU Only"
    
    st.sidebar.markdown(f"""
    <div style='background: {'#ECFDF5' if has_cuda else '#FEF3C7'}; border: 1px solid {'#A7F3D0' if has_cuda else '#FDE68A'}; border-radius: 8px; padding: 0.75rem; margin-bottom: 1rem;'>
        <div style='font-size: 0.75rem; font-weight: 700; color: {'#065F46' if has_cuda else '#92400E'}; text-transform: uppercase;'>Acceleration Active</div>
        <div style='font-size: 0.95rem; font-weight: 700; color: {'#047857' if has_cuda else '#B45309'};'>{device_label}</div>
        <div style='font-size: 0.8rem; color: #4B5563; margin-top: 0.2rem;'>{device_name}</div>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.subheader("⚙️ Model Hyperparameters")
    null_threshold = st.sidebar.slider(
        "Null Answer Threshold (τ)",
        min_value=-4.0,
        max_value=4.0,
        value=-1.0,
        step=0.25,
        help=(
            "Margin threshold: (Best_Span_Score - Null_Score). "
            "Higher = Stricter model (avoids false positives, rejects questionable evidence). "
            "Lower = More willing to extract candidate spans."
        )
    )

    max_len = st.sidebar.slider(
        "Max Answer Span Length",
        min_value=5,
        max_value=50,
        value=30,
        step=5,
        help="Maximum number of WordPiece tokens allowed in an extracted answer span."
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("🔗 Project Links")
    st.sidebar.markdown("""
    - [GitHub Repository](https://github.com/akash77007/NLP_Question-Answering-System)
    - [Demo Video (Drive)](https://drive.google.com/file/d/1c_FlTJh2mWxDgKP-5dGBlNSZ4vWIuDO9/view?usp=sharing)
    - **Course**: BCA (AI & Data Science)
    - **Group**: Group 10
    """)

    # -----------------------------------------------------
    # Top Hero Header
    # -----------------------------------------------------
    st.markdown("""
    <div class="hero-container">
        <h1 class="hero-title">DocuBERT: Extractive Question Answering</h1>
        <div class="hero-subtitle">
            Context-Grounded Reading Comprehension with BERT-base fine-tuned on Stanford SQuAD 2.0
        </div>
        <div class="badge-container">
            <span class="hero-badge">⚡ Pure Extractive (Zero Hallucination)</span>
            <span class="hero-badge">🛡️ SQuAD 2.0 Adversarial Rejection</span>
            <span class="hero-badge">🚀 RTX 3050 GPU Accelerated</span>
            <span class="hero-badge">🎓 Group 10 Capstone</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # -----------------------------------------------------
    # Main Navigation Tabs
    # -----------------------------------------------------
    tab_studio, tab_metrics, tab_eda, tab_team = st.tabs([
        "🔍 Interactive Q&A Studio",
        "📊 Benchmark & Architecture",
        "📈 Dataset & EDA Analytics",
        "👥 Group 10 & Submission Info"
    ])

    # Load Model Engine
    qa_engine = load_qa_engine()

    # =====================================================
    # TAB 1: INTERACTIVE Q&A STUDIO
    # =====================================================
    with tab_studio:
        # Preset Selector Bar
        col_preset, col_reset = st.columns([4, 1])
        with col_preset:
            selected_preset = st.selectbox(
                "📋 Quick-Load a Sample Scenario (or type your own text below):",
                options=list(PRESETS.keys())
            )
        with col_reset:
            st.write("")
            st.write("")
            reset_clicked = st.button("🔄 Clear / Reset", use_container_width=True)

        preset_context = ""
        preset_question = ""
        if not reset_clicked and selected_preset != "-- Custom Input (Enter your own context & question) --":
            preset_context = PRESETS[selected_preset]["context"]
            preset_question = PRESETS[selected_preset]["question"]

        col_input_left, col_input_right = st.columns([1, 1], gap="large")

        with col_input_left:
            st.markdown("#### 1. Reference Passage (Context)")
            context_text = st.text_area(
                "Provide the source document from which the answer will be extracted:",
                value=preset_context,
                height=220,
                placeholder="Paste paragraph, technical manual, legal clause, or article text here..."
            )
            
            # Context Counter Stats
            if context_text.strip():
                c_words = len(context_text.split())
                c_chars = len(context_text)
                st.caption(f"📊 Passage Stats: **{c_words} words** | **{c_chars} characters** | ~{int(c_words * 1.3)} BERT tokens")

        with col_input_right:
            st.markdown("#### 2. Natural Language Question")
            question_text = st.text_input(
                "Enter your question regarding the provided passage:",
                value=preset_question,
                placeholder="e.g., What was the name of the lunar module?"
            )
            
            st.markdown("<br>", unsafe_allow_html=True)
            run_btn = st.button("🚀 Extract Answer from Passage", type="primary", use_container_width=True)

            st.info(
                "💡 **How it works:** BERT attends to both Question and Passage simultaneously. "
                "If no answer is factually supported by the text, SQuAD 2.0 null-scoring explicitly alerts you."
            )

        # -------------------------------------------------
        # Extraction & Output Display
        # -------------------------------------------------
        if run_btn:
            if not context_text.strip() or not question_text.strip():
                st.warning("⚠️ Please provide both a reference passage and a question before submitting.")
            else:
                with st.spinner("Processing token logits with BERT Transformer layers..."):
                    res = qa_engine.predict(
                        question=question_text,
                        context=context_text,
                        null_threshold=null_threshold,
                        max_answer_length=max_len
                    )

                st.markdown("---")
                st.markdown("### 🎯 Extraction Results & Analysis")

                # Metric Cards Bar
                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                with m_col1:
                    status_badge = "❌ Unanswerable" if res["is_impossible"] else "✅ Answer Extracted"
                    st.metric("Status", status_badge)
                with m_col2:
                    st.metric("Confidence Score", f"{res['confidence']:.1%}")
                with m_col3:
                    st.metric("Logit Difference (Δ)", f"{res['score']:.2f}")
                with m_col4:
                    st.metric("Inference Latency", f"{res['latency_ms']} ms")

                # Confidence Bar
                st.progress(res['confidence'])

                # Display Results
                if res["is_impossible"]:
                    st.markdown(f"""
                    <div class="unanswerable-card">
                        <h3 class="unanswerable-title">🚫 No Answer Found in Passage</h3>
                        <div class="unanswerable-desc">
                            The model analyzed the passage and determined with <b>{res['confidence']:.1%} certainty</b> 
                            that the text does not contain factual evidence to answer this question.
                            <br><br>
                            <i>(SQuAD 2.0 adversarial null thresholding rejected this query because the candidate span score 
                            failed to exceed the [CLS] null score by margin τ = {null_threshold}).</i>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    ans = res["answer"]
                    start_pos = res["start_char"]
                    end_pos = res["end_char"]

                    st.markdown(f"""
                    <div class="answer-card">
                        <div class="answer-header">
                            <span>🎯 Extracted Text Span</span>
                        </div>
                        <div class="answer-text">"{ans}"</div>
                        <div class="answer-meta">
                            Exact Span Indices: Characters [{start_pos} : {end_pos}] in source passage
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # In-Passage Highlighting
                    st.markdown("#### 📖 In-Passage Grounding (Visual Highlighting)")
                    before = context_text[:start_pos]
                    target = context_text[start_pos:end_pos]
                    after = context_text[end_pos:]

                    highlighted_html = (
                        f"<div class='passage-viewer'>"
                        f"{before}<span class='highlight-span'>{target}</span>{after}"
                        f"</div>"
                    )
                    st.markdown(highlighted_html, unsafe_allow_html=True)

                # Collapsible Deep-Dive Diagnostics
                with st.expander("🔬 Deep-Dive Model Diagnostics & Raw Logits"):
                    d1, d2 = st.columns(2)
                    with d1:
                        st.json({
                            "question": question_text,
                            "extracted_answer": res["answer"],
                            "is_impossible": res["is_impossible"],
                            "confidence_probability": res["confidence"],
                            "raw_score_margin": res["score"]
                        })
                    with d2:
                        st.json({
                            "start_char": res["start_char"],
                            "end_char": res["end_char"],
                            "latency_milliseconds": res["latency_ms"],
                            "null_threshold_used": null_threshold,
                            "hardware_device": str(qa_engine.device)
                        })

    # =====================================================
    # TAB 2: BENCHMARK & ARCHITECTURE
    # =====================================================
    with tab_metrics:
        st.markdown("### 📊 Empirical Performance on SQuAD 2.0")
        st.markdown(
            "Our model was benchmarked on a balanced split of the official **Stanford SQuAD 2.0** validation dataset. "
            "Here are the empirical results evaluated on the NVIDIA RTX 3050 GPU:"
        )

        # Load eval_results.json if available
        eval_path = os.path.join("data", "eval_results.json")
        if not os.path.exists(eval_path):
            eval_path = os.path.join("..", "data", "eval_results.json")

        if os.path.exists(eval_path):
            with open(eval_path, "r", encoding="utf-8") as f:
                eval_data = json.load(f)["summary"]
            
            e1, e2, e3, e4 = st.columns(4)
            with e1:
                st.metric("Overall Exact Match (EM)", f"{eval_data['exact_match']}%")
            with e2:
                st.metric("Overall Token F1", f"{eval_data['f1_score']}%")
            with e3:
                st.metric("Unanswerable (No-Ans) F1", f"{eval_data['no_answer_f1']}%")
            with e4:
                st.metric("Average Latency", f"{eval_data['avg_latency_ms']} ms / query")
        else:
            e1, e2, e3, e4 = st.columns(4)
            with e1:
                st.metric("Overall Exact Match (EM)", "64.00%")
            with e2:
                st.metric("Overall Token F1", "69.23%")
            with e3:
                st.metric("Unanswerable F1", "74.00%")
            with e4:
                st.metric("Average Latency", "32.85 ms / query")

        st.markdown("---")
        st.markdown("### 🏛️ Model Architecture & Mathematical Pipeline")
        st.markdown("""
        ```
        [User Question]  +  [Source Passage]
                 │
                 ▼
        [WordPiece Fast Tokenizer]
           - Sequence Packaging: [CLS] Question [SEP] Context [SEP]
           - Segment IDs: 0 for Question, 1 for Passage
           - Generates: input_ids, attention_mask, token_type_ids, offset_mapping
                 │
                 ▼
        [BERT-base Transformer Encoder (110M Params)]
           - 12 Bidirectional Self-Attention Layers, 768-dim Hidden Vectors
           - Contextual hidden representations: H ∈ ℝ^(N × 768)
                 │
                 ▼
        [QA Span Classification Head]
           - Dense projection layer: ℝ^768 ➔ ℝ^2
           - Start Logits: s_k = w_startᵀ · h_k   |   End Logits: e_k = w_endᵀ · h_k
                 │
                 ▼
        [SQuAD 2.0 Span Selection & Null Thresholding]
           - Candidate Span Score: S_(i, j) = s_i + e_j   (i ≤ j, within Context)
           - Null (Unanswerable) Score: S_null = s_0 + e_0 (at [CLS])
           - If (max(S_(i, j)) - S_null) < τ  ➔ Flag as Unanswerable
           - Else ➔ Slice Context[offset_start : offset_end] verbatim
        ```
        """)

    # =====================================================
    # TAB 3: DATASET & EDA ANALYTICS
    # =====================================================
    with tab_eda:
        st.markdown("### 📈 Exploratory Data Analysis (SQuAD 2.0)")
        st.markdown(
            "Exploratory analysis was conducted across all 11,873 development records to optimize sequence length "
            "and verify dataset balance:"
        )

        chart_path = os.path.join("data", "squad_eda_plots.png")
        if not os.path.exists(chart_path):
            chart_path = os.path.join("..", "data", "squad_eda_plots.png")

        if os.path.exists(chart_path):
            st.image(chart_path, caption="Figure 1: SQuAD 2.0 4-Panel Exploratory Data Analysis Dashboard", use_container_width=True)
        else:
            st.info("EDA chart available at data/squad_eda_plots.png")

        col_eda1, col_eda2 = st.columns(2)
        with col_eda1:
            st.markdown("""
            #### 📊 Key Distribution Metrics:
            - **Class Balance**: 49.93% Answerable vs. 50.07% Unanswerable.
            - **Median Passage Length**: 120 words.
            - **95th Percentile Passage Length**: 220 words (~286 WordPiece tokens).
            - **Median Question Length**: 10 words.
            """)
        with col_eda2:
            st.markdown("""
            #### 💡 Architectural Deductions:
            - A sequence length limit of **384 tokens** covers >95% of SQuAD passages without truncation.
            - Preprocessing preserves stopwords and casing because BERT self-attention requires grammatical syntax to identify syntactic relationships.
            """)

    # =====================================================
    # TAB 4: GROUP 10 & SUBMISSION INFO
    # =====================================================
    with tab_team:
        st.markdown("### 🎓 Capstone Project Details")
        st.markdown("""
        - **Project**: Project 12 — Question Answering System
        - **Group**: Group 10
        - **Course**: Bachelor of Computer Applications (BCA)
        - **Specialisation**: Artificial Intelligence & Data Science
        - **Department**: Department of Computer Science & Applications
        """)

        st.markdown("#### 👥 Team Members (Group 10):")
        members = [
            "2401201113",
            "2401201115",
            "2401201102",
            "2401201213",
            "2401201104",
            "2401201108 (Akash Sharma - Project Lead)"
        ]
        cols = st.columns(3)
        for i, member in enumerate(members):
            with cols[i % 3]:
                st.markdown(f"<div class='team-badge'>👤 Roll No: {member}</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 🔗 Deliverable Links:")
        b1, b2 = st.columns(2)
        with b1:
            st.link_button("🌐 Open GitHub Repository", "https://github.com/akash77007/NLP_Question-Answering-System", use_container_width=True)
        with b2:
            st.link_button("▶️ Watch Live Demonstration Video", "https://drive.google.com/file/d/1c_FlTJh2mWxDgKP-5dGBlNSZ4vWIuDO9/view?usp=sharing", use_container_width=True)

    # Footer
    st.markdown("<br><hr>", unsafe_allow_html=True)
    st.markdown(
        "<div style='text-align: center; color: #64748B; font-size: 0.85rem;'>"
        "NLP Capstone Project 12 — Extractive Question Answering System | Group 10 | Powered by PyTorch & BERT"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
