# Capstone Project Report

## Project 12: Extractive Question Answering System Using BERT and SQuAD 2.0

**Course:** Bachelor of Computer Applications (BCA)  
**Specialisation:** Artificial Intelligence and Data Science  
**Subject:** Natural Language Processing (NLP) Capstone Project  
**Group Number:** Group 10  
**GitHub Repository:** [https://github.com/akash77007/NLP_Question-Answering-System](https://github.com/akash77007/NLP_Question-Answering-System)  

**Team Members (Group 10):**
1. Roll No: 2401201113
2. Roll No: 2401201115
3. Roll No: 2401201102
4. Roll No: 2401201213
5. Roll No: 2401201104
6. Roll No: 2401201108 (Akash Sharma)  

---

## Abstract

Extractive Question Answering (QA) is a fundamental Natural Language Processing task wherein a computational model reads an unstructured passage of text and extracts the exact sub-sequence (span) that answers a given user query. While modern generative large language models (LLMs) synthesize fluent responses, they suffer from factual hallucination and unverified assertions. In contrast, extractive question answering enforces strict factual grounding by restricting all answers to verbatim spans present in the source document.

This project designs, implements, evaluates, and deploys an end-to-end Extractive Question Answering System based on the Bidirectional Encoder Representations from Transformers (BERT) architecture fine-tuned on the Stanford Question Answering Dataset 2.0 (SQuAD 2.0). The system implements an adversarial null-thresholding mechanism to detect unanswerable queries, an offset-mapping module for verbatim character span extraction, and an interactive, GPU-accelerated web graphical user interface (GUI) developed in Streamlit. Benchmarked on the SQuAD 2.0 development set using an NVIDIA GeForce RTX 3050 Laptop GPU, the system achieves an overall Exact Match (EM) of 64.00%, a token-level F1 score of 69.23%, an unanswerable detection accuracy of 74.00%, and an average inference latency of 32.85 ms per query.

---

## 1. Introduction and Objectives

### 1.1 Problem Statement
In information-dense domains such as legal discovery, clinical healthcare, and enterprise compliance, users require precise answers grounded solely in provided source documents. Generative models often generate plausible-sounding falsehoods (hallucinations) when confronted with questions outside their training data or questions whose answers are absent from the context. 

Project 12 mandates the development of an **extractive** Question Answering system that:
1. Operates on a user-supplied context passage and a natural language question.
2. Identifies and extracts the exact answer substring directly from the passage.
3. Detects when a question is unanswerable from the passage and explicitly notifies the user.
4. Provides a demonstrable web graphical user interface (GUI).

### 1.2 Key Objectives
- **Strict Factual Grounding:** Eliminate hallucination by restricting answers to contiguous passage spans.
- **Adversarial Robustness:** Incorporate SQuAD 2.0 null-score rejection to reliably handle questions where evidence is missing.
- **Hardware-Accelerated Inference:** Leverage PyTorch and CUDA to run real-time inference on consumer GPU hardware (NVIDIA RTX 3050, 4 GB VRAM).
- **Interactive Visualization:** Build a Streamlit application featuring live in-passage highlighting, confidence metrics, and sensitivity controls.

---

## 2. Dataset Description: Stanford SQuAD 2.0

### 2.1 Overview
The Stanford Question Answering Dataset (SQuAD) is the industry-standard benchmark for reading comprehension. While SQuAD 1.1 contained over 100,000 answerable question-passage pairs, it suffered from a major flaw: models were rewarded for guessing spans even on nonsensical questions.

SQuAD 2.0 addresses this vulnerability by combining the 100,000 answerable questions from SQuAD 1.1 with over **50,000 unanswerable questions** written adversarially by human crowdworkers. These unanswerable questions share key nouns and verbs with the passage to mislead naive matching algorithms, forcing the model to verify semantic entails before extracting an answer.

### 2.2 Dataset Schema
Each entry in the SQuAD 2.0 tabular split consists of:
- `title`: The Wikipedia article from which the context was sourced.
- `context`: The reference passage text.
- `question`: The natural language query.
- `ids`: Unique hexadecimal identifier.
- `answers`: Ground-truth human reference answers.
- `answer_start`: Character index where each reference answer begins in the context.
- `is_impossible`: Boolean flag (`False` = Answerable, `True` = Unanswerable).

---

## 3. Exploratory Data Analysis (EDA)

Exploratory analysis was conducted using Pandas and Matplotlib across the 11,873 samples of the SQuAD 2.0 validation split. The findings are summarized below:

### 3.1 Class Balance
- **Answerable Questions (`is_impossible == False`):** 5,928 (49.93%)
- **Unanswerable Questions (`is_impossible == True`):** 5,945 (50.07%)
- *Insight:* The validation split is precisely balanced 50/50. A baseline model that uniformly guesses "unanswerable" would achieve 50% accuracy, making dual evaluation on both Answerable and Unanswerable subsets mandatory.

### 3.2 Sequence Length Distributions
- **Passage Length (Words):**
  - Median: 120 words
  - 90th Percentile: 194 words
  - 95th Percentile: 220 words
  - Maximum: 629 words
- **Question Length (Words):**
  - Median: 10 words
  - 95th Percentile: 15 words
- *Architectural Implication:* With an average token-to-word ratio of approximately 1.3 for WordPiece subwords, a passage of 220 words translates to ~286 tokens. Adding a 15-word question (~20 tokens) and special tokens (`[CLS]`, `[SEP]`, `[SEP]`) yields ~309 tokens, which fits comfortably within BERT's standard `max_seq_length = 384` window without truncating critical context.

### 3.3 Linguistic Question Typology
Analysis of the leading interrogative words revealed the following distribution:
1. **What:** 58.2%
2. **Who:** 14.1%
3. **How:** 10.3%
4. **When:** 9.1%
5. **Where:** 5.4%
6. **Why / Which:** 2.9%

*(Reference: See generated 4-panel figure at `data/squad_eda_plots.png`)*

---

## 4. System Architecture and Methodology

### 4.1 Model Selection
The system employs **`deepset/bert-base-cased-squad2`**, a 12-layer, 768-hidden dimension BERT-base Transformer model (110 million parameters) fine-tuned specifically on SQuAD 2.0. The cased model preserves capitalization, which is essential for identifying named entities (e.g., distinguishing "apple" the fruit from "Apple" the corporation).

### 4.2 Mathematical Span Formulation

#### 4.2.1 Input Formatting
Given question tokens $Q = (q_1, \dots, q_m)$ and context tokens $C = (c_1, \dots, c_n)$, the sequence is structured as:
$$\mathbf{X} = [\text{CLS}], q_1, \dots, q_m, [\text{SEP}], c_1, \dots, c_n, [\text{SEP}]$$

Segment embeddings assign $0$ to all tokens up to the first $[\text{SEP}]$, and $1$ to the context tokens.

#### 4.2.2 Logit Computation
BERT outputs contextual hidden vectors $\mathbf{h}_k \in \mathbb{R}^{768}$ for each position $k$. A linear classification layer with trainable weights $\mathbf{w}_s, \mathbf{w}_e \in \mathbb{R}^{768}$ computes start and end logits:
$$s_k = \mathbf{w}_s^\top \mathbf{h}_k \quad (\text{Start Logit})$$
$$e_k = \mathbf{w}_e^\top \mathbf{h}_k \quad (\text{End Logit})$$

#### 4.2.3 Span Candidate Selection
The system filters candidate spans $(i, j)$ such that:
1. $i \le j$ (the answer end must follow or equal the answer start).
2. $j - i + 1 \le L_{\max}$ (constrained to $L_{\max} = 30$ tokens).
3. Both $i$ and $j$ reside within the context passage (excluding question and special tokens).

The span score is defined as:
$$S_{(i, j)} = s_i + e_j$$

#### 4.2.4 Unanswerable Detection (Null Thresholding)
In SQuAD 2.0, unanswerable questions are supervised during fine-tuning with the $[\text{CLS}]$ token (position 0) as the target. The null score is computed as:
$$S_{\text{null}} = s_0 + e_0$$

The difference margin is evaluated:
$$\Delta = \max_{(i, j)} S_{(i, j)} - S_{\text{null}}$$

If $\Delta < \tau$ (where $\tau = -1.0$ is the calibrated null threshold), the model concludes that the passage contains insufficient evidence and outputs:
$$\text{Answer} = \text{"No answer found in the provided passage."}$$

#### 4.2.5 Lossless Character Slicing
To avoid WordPiece detokenization artifacts (such as missing spaces or mangled punctuation), the Hugging Face fast Rust tokenizer generates an `offset_mapping` tensor containing the exact character boundaries $(c_{\text{start}}, c_{\text{end}})$ in the raw input string. Slicing `context[c_start:c_end]` guarantees verbatim text recovery with original formatting intact.

---

## 5. Experimental Evaluation and Results

The evaluation pipeline was executed on a balanced sample of the SQuAD 2.0 development set using an NVIDIA RTX 3050 Laptop GPU.

### 5.1 Metrics
- **Exact Match (EM):** Binary metric (1 or 0). After lowercasing, stripping punctuation, and removing English articles ('a', 'an', 'the'), the prediction must match the ground truth identically.
- **Token-Level F1 Score:** Harmonic mean of precision and recall over word tokens:
  $$\text{Precision} = \frac{|\text{Tokens}_{\text{pred}} \cap \text{Tokens}_{\text{gold}}|}{|\text{Tokens}_{\text{pred}}|}, \quad \text{Recall} = \frac{|\text{Tokens}_{\text{pred}} \cap \text{Tokens}_{\text{gold}}|}{|\text{Tokens}_{\text{gold}}|}$$
  $$\text{F1} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

### 5.2 Benchmark Results Table

| Evaluation Category | Sample Count | Exact Match (EM) | F1 Score |
| :--- | :---: | :---: | :---: |
| **Overall Dataset** | **100** | **64.00%** | **69.23%** |
| **Answerable Questions (Has-Ans)** | 50 | 54.00% | 64.46% |
| **Unanswerable Questions (No-Ans)** | 50 | 74.00% | 74.00% |
| **Average Inference Latency** | — | — | **32.85 ms / query** |

### 5.3 Error Analysis
1. **Span Boundary Sensitivity:** Several answerable errors obtained $\text{EM} = 0\%$ but $\text{F1} > 80\%$. For example, when asked for the EU judicial body, the model extracted *"Court of Justice of the European Union"* while the human annotator wrote *"the Court of Justice of the European Union (CJEU)"*.
2. **Conservative Null Rejection:** With $\tau = -1.0$, the model rejected 74% of unanswerable questions successfully. Adjusting $\tau$ higher increases precision at the expense of recall on borderline answerable questions.

---

## 6. Graphical User Interface (GUI) Design

The user interface is implemented in Streamlit (`app.py`), following clean Human-Computer Interaction (HCI) principles:

1. **Passage Input & Word Counter:** Large text area displaying real-time character and word metrics.
2. **Preset Selector:** One-click dropdown loading curated scenarios (Apollo 11, Photosynthesis, Alan Turing, Eiffel Tower adversarial trick).
3. **In-Passage Visual Highlighter:** Highlights the exact extracted answer in yellow directly inside the rendered passage.
4. **Metrics Dashboard:** Displays Extraction Status, Calibrated Confidence %, Logit Margin ($\Delta$), and Latency in milliseconds.
5. **Sidebar Sensitivity Slider:** Allows dynamic adjustment of the null threshold $\tau$ during live demonstrations.
6. **Resource Caching:** Employs `@st.cache_resource` to load the BERT checkpoint into GPU VRAM once, guaranteeing sub-50ms subsequent queries.

---

## 7. Conclusion and Future Work

### 7.1 Conclusion
Project 12 has been successfully completed. The extractive QA pipeline provides:
- Verifiable, hallucination-free answers grounded strictly in user documents.
- Effective rejection of unanswerable and adversarial queries via SQuAD 2.0 null scoring.
- Real-time performance (~30 ms/query) on laptop hardware.
- A functional, polished web application satisfying all college capstone guidelines.

### 7.2 Future Enhancements
1. **Retrieval-Augmented Generation (RAG):** Integrate a dense vector retriever (e.g., FAISS with BGE-embeddings) to scale from single passages to multi-document corporate corpora.
2. **Model Quantization:** Quantize weights to INT8 using ONNX Runtime or TensorRT to reduce memory consumption below 200 MB.
3. **Multilingual Adaptation:** Upgrade the encoder to `xlm-roberta-base` to support multilingual extractive QA across Indian vernacular languages.

---


