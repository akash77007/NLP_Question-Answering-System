"""
app.py
Streamlit Web Application for Project 12: Extractive Question Answering System.
Built with PyTorch, Hugging Face Transformers (BERT SQuAD 2.0), and Streamlit.
"""

import time
import os
import json
import streamlit as st
import torch
from src.inference import QuestionAnsweringEngine

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="BERT Question Answering System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean UI styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .answer-box {
        background-color: #ECFDF5;
        border-left: 5px solid #10B981;
        padding: 1.2rem;
        border-radius: 8px;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .unanswerable-box {
        background-color: #FEF2F2;
        border-left: 5px solid #EF4444;
        padding: 1.2rem;
        border-radius: 8px;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .highlight-span {
        background-color: #FEF08A;
        color: #854D0E;
        font-weight: 700;
        padding: 2px 6px;
        border-radius: 4px;
        border: 1px solid #FACC15;
    }
    .metric-card {
        background: #F3F4F6;
        padding: 0.8rem;
        border-radius: 6px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Cached Model Loader (loads once into GPU memory)
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Loading fine-tuned BERT QA model into memory...")
def load_qa_engine():
    """Load model once and keep in memory across user interactions."""
    return QuestionAnsweringEngine(model_name="deepset/bert-base-cased-squad2")


# ---------------------------------------------------------
# Presets for Quick Testing & Viva Demo
# ---------------------------------------------------------
PRESETS = {
    "-- Select a Sample Passage or Type Your Own --": {
        "context": "",
        "question": ""
    },
    "Apollo 11 Moon Landing (Answerable)": {
        "context": (
            "Apollo 11 was the American spaceflight that first landed humans on the Moon. "
            "Commander Neil Armstrong and Lunar Module pilot Buzz Aldrin landed the Apollo Lunar Module Eagle "
            "on July 20, 1969, at 20:17 UTC, and Armstrong became the first person to step onto the Moon's surface "
            "six hours and 39 minutes later, on July 21 at 02:56 UTC. Aldrin joined him 19 minutes later. "
            "They spent about two and a quarter hours together exploring the site they had named Tranquility Base."
        ),
        "question": "What was the name of the lunar module that landed on the Moon?"
    },
    "Photosynthesis in Plants (Answerable)": {
        "context": (
            "Photosynthesis is a biological process used by plants and other organisms to convert light energy "
            "into chemical energy that can later be released to fuel the organism's activities. This chemical "
            "energy is stored in carbohydrate molecules, such as sugars, which are synthesized from carbon dioxide "
            "and water. In most cases, oxygen is also released as a waste product."
        ),
        "question": "What waste product is released during photosynthesis?"
    },
    "Artificial Intelligence (Answerable)": {
        "context": (
            "Alan Turing was an English mathematician, computer scientist, logician, and theoretical biologist. "
            "He was highly influential in the development of theoretical computer science, providing a formalisation "
            "of the concepts of algorithm and computation with the Turing machine, which can be considered a model "
            "of a general-purpose computer. In 1950, Turing proposed the Turing test as a measure of machine intelligence."
        ),
        "question": "When did Turing propose the Turing test?"
    },
    "Adversarial Trick Question (Unanswerable / SQuAD 2.0 Test)": {
        "context": (
            "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. "
            "It is named after the engineer Gustave Eiffel, whose company designed and built the tower. "
            "Constructed from 1887 to 1889 as the entrance arch to the 1889 World's Fair, it was initially "
            "criticized by some of France's leading artists and intellectuals for its design."
        ),
        "question": "How many kilograms of gold were used to paint the Eiffel Tower?"
    }
}


def main():
    # Sidebar
    st.sidebar.title("⚙️ Model Settings")
    
    device_name = "NVIDIA CUDA GPU" if torch.cuda.is_available() else "CPU"
    gpu_detail = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Host CPU"
    
    st.sidebar.success(f"**Hardware Device:** {device_name}\n\n*{gpu_detail}*")
    st.sidebar.markdown("---")

    st.sidebar.subheader("Confidence & Thresholding")
    null_threshold = st.sidebar.slider(
        "Null Answer Threshold",
        min_value=-5.0,
        max_value=5.0,
        value=-1.0,
        step=0.25,
        help=(
            "Controls the sensitivity for unanswerable questions. "
            "Higher threshold = stricter model (requires stronger evidence before answering). "
            "Lower threshold = more willing to extract answer spans."
        )
    )
    
    max_len = st.sidebar.slider(
        "Max Answer Length (Tokens)",
        min_value=5,
        max_value=60,
        value=30,
        step=5
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 System Specs")
    st.sidebar.info(
        "**Model**: BERT-base (Cased)\n\n"
        "**Architecture**: 12 Layers, 768 Hidden, 12 Heads (110M params)\n\n"
        "**Trained On**: Stanford SQuAD 2.0\n\n"
        "**Extraction Type**: Pure Extractive (Token Logits)"
    )

    # Main Header
    st.markdown("<div class='main-header'>Extractive Question Answering System</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-header'>Project 12 — College NLP Capstone | Powered by BERT & SQuAD 2.0</div>",
        unsafe_allow_html=True
    )

    # Initialize Engine
    qa_engine = load_qa_engine()

    # Preset selection
    col_preset, col_clear = st.columns([4, 1])
    with col_preset:
        selected_preset = st.selectbox(
            "📋 Choose a sample passage & question (or enter your own below):",
            options=list(PRESETS.keys())
        )
    with col_clear:
        st.write("")
        st.write("")
        clear_clicked = st.button("🔄 Reset Inputs", use_container_width=True)

    default_context = ""
    default_question = ""
    if not clear_clicked and selected_preset != "-- Select a Sample Passage or Type Your Own --":
        default_context = PRESETS[selected_preset]["context"]
        default_question = PRESETS[selected_preset]["question"]

    # Input forms
    st.markdown("### 1. Source Passage (Context)")
    context_input = st.text_area(
        "Enter the reference document or passage that the system must answer from:",
        value=default_context,
        height=180,
        placeholder="Paste your paragraph or article text here..."
    )

    # Passage stats
    if context_input.strip():
        word_count = len(context_input.split())
        char_count = len(context_input)
        st.caption(f"📝 Passage length: **{word_count} words** | **{char_count} characters**")

    st.markdown("### 2. Question")
    question_input = st.text_input(
        "Enter your question regarding the passage above:",
        value=default_question,
        placeholder="e.g., What was the name of the lunar module?"
    )

    st.markdown("<br>", unsafe_allow_html=True)
    get_answer_btn = st.button("🔍 Get Answer", type="primary", use_container_width=True)

    # Prediction Execution
    if get_answer_btn:
        if not context_input.strip() or not question_input.strip():
            st.warning("⚠️ Please provide both a passage and a question before submitting.")
        else:
            with st.spinner("Analyzing text with BERT attention layers..."):
                res = qa_engine.predict(
                    question=question_input,
                    context=context_input,
                    null_threshold=null_threshold,
                    max_answer_length=max_len
                )

            st.markdown("---")
            st.markdown("### 🎯 Extraction Results")

            # Metric Cards
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                status_text = "❌ Unanswerable" if res["is_impossible"] else "✅ Answer Found"
                st.metric("Status", status_text)
            with m2:
                st.metric("Model Confidence", f"{res['confidence']:.1%}")
            with m3:
                st.metric("Logit Margin (Δ)", f"{res['score']:.2f}")
            with m4:
                st.metric("Inference Time", f"{res['latency_ms']} ms")

            # Result Display
            if res["is_impossible"]:
                st.markdown(f"""
                <div class='unanswerable-box'>
                    <h4 style='color: #991B1B; margin: 0;'>🚫 No Answer Found in Passage</h4>
                    <p style='color: #4B5563; margin-top: 0.5rem;'>
                        The model determined with <b>{res['confidence']:.1%} certainty</b> that the provided passage 
                        does not contain factual evidence to answer this question.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                extracted = res["answer"]
                start_c = res["start_char"]
                end_c = res["end_char"]

                st.markdown(f"""
                <div class='answer-box'>
                    <span style='font-size: 0.85rem; font-weight: 600; color: #065F46; text-transform: uppercase;'>Extracted Answer Span:</span>
                    <h2 style='color: #065F46; margin-top: 0.2rem; margin-bottom: 0.4rem;'>"{extracted}"</h2>
                    <span style='font-size: 0.85rem; color: #047857;'>Characters [{start_c} : {end_c}] in context</span>
                </div>
                """, unsafe_allow_html=True)

                # Visual Passage Highlight
                st.markdown("#### 📖 Context with Highlighted Answer:")
                before = context_input[:start_c]
                target = context_input[start_c:end_c]
                after = context_input[end_c:]

                highlighted_html = (
                    f"<div style='background-color: #F9FAFB; padding: 1.2rem; border-radius: 8px; line-height: 1.7; border: 1px solid #E5E7EB;'>"
                    f"{before}<span class='highlight-span'>{target}</span>{after}"
                    f"</div>"
                )
                st.markdown(highlighted_html, unsafe_allow_html=True)

    # Footer
    st.markdown("<br><hr>", unsafe_allow_html=True)
    st.caption("College Capstone Project 12 — NLP Question Answering System | SQuAD 2.0 Extractive BERT Engine")


if __name__ == "__main__":
    main()
