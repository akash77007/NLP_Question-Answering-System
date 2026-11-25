# Capstone Project Report

## Project 12: Extractive Question Answering System Using BERT and SQuAD 2.0

**Course:** Bachelor of Computer Applications (BCA)  
**Specialisation:** Artificial Intelligence and Data Science  
**Subject:** Natural Language Processing (NLP) Capstone Project  
**Group Number:** Group 10  
**GitHub Repository:** [https://github.com/akash77007/NLP_Question-Answering-System](https://github.com/akash77007/NLP_Question-Answering-System)  
**Project Demo Video:** [https://drive.google.com/file/d/1c_FlTJh2mWxDgKP-5dGBlNSZ4vWIuDO9/view?usp=sharing](https://drive.google.com/file/d/1c_FlTJh2mWxDgKP-5dGBlNSZ4vWIuDO9/view?usp=sharing)  

**Team Members (Group 10):**
1. **Akash Sharma (Project Lead)** — Roll No: `2401201108`
2. **Khushi** — Roll No: `2401201102`
3. **Aryan Sareen** — Roll No: `2401201104`
4. **Aman** — Roll No: `2401201115`
5. **Vivek Sharma** — Roll No: `2401201213`

---

## Abstract

Extractive Question Answering (QA) is an essential branch of Natural Language Processing (NLP) in which an automated model reads an unstructured passage of text and extracts the precise sub-sequence (text span) that answers a user's question. While modern generative Large Language Models (LLMs) synthesize fluent answers, they suffer from factual hallucination, unverified assertions, and high computational cost. In mission-critical domains—such as legal contract discovery, clinical healthcare, and regulatory financial compliance—answers must be strictly grounded in provided reference text with verifiable provenance.

This project designs, evaluates, and deploys an end-to-end, production-grade Extractive Question Answering System based on the Bidirectional Encoder Representations from Transformers (BERT) architecture fine-tuned on the Stanford Question Answering Dataset 2.0 (SQuAD 2.0). The system addresses key reading comprehension challenges by incorporating:
1. Pure extractive span extraction predicting start and end token logits.
2. Adversarial unanswerable question detection using [CLS] null-thresholding, enabling the model to explicitly reject trick questions lacking supporting evidence.
3. Lossless character offset mapping that recovers verbatim passage text with original casing, punctuation, and whitespace intact.
4. An interactive, GPU-accelerated web graphical user interface (GUI) developed in Streamlit featuring real-time in-passage HTML visual highlighting, confidence scoring, and sensitivity threshold sliders.

Benchmarked on a balanced split of the SQuAD 2.0 validation dataset using an NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM), the system achieves an overall Exact Match (EM) score of 64.00%, an overall token-level F1 score of 69.23%, an unanswerable question accuracy of 74.00%, and an average inference latency of 32.85 milliseconds per query (~30 queries per second). The resulting application fully satisfies all college capstone requirements, providing an auditable, high-performance question answering tool with zero risk of generative hallucination.

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
- **Answerable Questions (`is_impossible == False`):** 5,928 samples (49.93%)
- **Unanswerable Questions (`is_impossible == True`):** 5,945 samples (50.07%)
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
1. **What:** 58.2% (Object, concept, definition)
2. **Who:** 14.1% (Person, organization, entity)
3. **How:** 10.3% (Manner, process, quantity)
4. **When:** 9.1% (Date, temporal events)
5. **Where:** 5.4% (Location, place)
6. **Why / Which:** 2.9% (Causal & selection)

![Figure 1: SQuAD 2.0 Exploratory Data Analysis Dashboard](../data/squad_eda_plots.png)
*Figure 1: Exploratory Data Analysis dashboard for SQuAD 2.0 validation split (n=11,873). Panel A shows the 50/50 class balance; Panel B plots context word counts with the 95th percentile cutoff at 220 words; Panel C displays question length distributions; Panel D highlights interrogative starter frequencies.*

### 3.4 Text Preprocessing & Offset Mapping
Unlike traditional NLP systems where text is normalized by removing stopwords, stemming, and stripping punctuation, BERT operates on full grammatical syntax:
- Stopwords provide necessary relational signals for bidirectional attention heads.
- Punctuation marks establish clause boundaries.
- Casing is critical to disambiguate proper nouns and entities.

Input sequences are tokenized using WordPiece with the joint formatting:
$$\mathbf{X} = [\text{CLS}], q_1, \dots, q_m, [\text{SEP}], c_1, \dots, c_n, [\text{SEP}]$$

The fast Rust tokenizer computes an `offset_mapping` tensor containing the start and end character offsets $(c_{\text{start}}, c_{\text{end}})$ of each token within the raw passage string, enabling direct verbatim slicing without detokenization artifacts.

---

## 4. System Architecture and Methodology

### 4.1 Model Selection
The system employs **`deepset/bert-base-cased-squad2`**, a 12-layer, 768-hidden dimension BERT-base Transformer model (110 million parameters) fine-tuned specifically on SQuAD 2.0. The cased model preserves capitalization, which is essential for identifying named entities.

![Figure 2: BERT Extractive Question Answering Architecture](screenshots/bert_qa_architecture.png)
*Figure 2: End-to-end architecture of the BERT Extractive Question Answering Engine. The model encodes joint Question-Passage sequences via 12 Transformer layers, projecting output contextual embeddings $T_i$ into start logits $s_i$ and end logits $e_j$, with [CLS] representation $T_0$ dedicated to SQuAD 2.0 adversarial null scoring.*

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

The user interface is implemented in Streamlit (`app.py`), styled with Google's Material You (Material Design 3) design system.

![Figure 3: Streamlit Web Application Interface](screenshots/02_answer_extracted_apollo.png)
*Figure 3: Streamlit Web GUI (Material You design system) executing extractive inference on an Apollo 11 historical passage. The verbatim answer ("Eagle") is surfaced with 99.4% confidence and illuminated directly inside the source document.*

![Figure 4: Adversarial Query Rejection](screenshots/03_adversarial_rejection_eiffel.png)
*Figure 4: Streamlit Web GUI rejecting an adversarial trick question lacking supporting evidence in the text ("How many kilograms of gold were used to paint the Eiffel Tower?"). SQuAD 2.0 null scoring triggers a high-certainty unanswerable rejection card with zero generative hallucination.*

![Figure 5: Model Diagnostics and Token Offsets](screenshots/04_diagnostics_expanded.png)
*Figure 5: Collapsible viva defense diagnostics panel exposing exact character start and end indices (offsets 111-116), logit score margins, and raw prediction JSON metadata for auditable extraction provenance.*

![Figure 6: Model Architecture & Benchmark Performance Dashboard](screenshots/05_architecture_benchmark.png)
*Figure 6: Empirical evaluation benchmark dashboard integrated into Tab 2 of the Streamlit application, displaying Exact Match (64.00%), F1 score (69.23%), and RTX 3050 GPU latency (32.85 ms).*

### 6.1 GUI Features
1. **Passage Input & Word Counter:** Large text area displaying real-time character and word metrics.
2. **Preset Selector:** One-click dropdown loading curated scenarios (Apollo 11, Photosynthesis, Alan Turing, Eiffel Tower adversarial trick, Commercial Contract clause).
3. **In-Passage Visual Highlighter:** Highlights the exact extracted answer in yellow directly inside the rendered passage.
4. **Metrics Dashboard:** Displays Extraction Status, Calibrated Confidence %, Logit Margin ($\Delta$), and Latency in milliseconds.
5. **Sidebar Sensitivity Slider:** Allows dynamic adjustment of the null threshold $\tau$ during live demonstrations.
6. **Resource Caching:** Employs `@st.cache_resource` to load the BERT checkpoint into GPU VRAM once, guaranteeing sub-50ms subsequent queries.
7. **Demonstration Video:** A complete walkthrough of the working application is available at: [https://drive.google.com/file/d/1c_FlTJh2mWxDgKP-5dGBlNSZ4vWIuDO9/view?usp=sharing](https://drive.google.com/file/d/1c_FlTJh2mWxDgKP-5dGBlNSZ4vWIuDO9/view?usp=sharing)

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

## 8. References

[1] Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." arXiv:1810.04805.  
[2] Rajpurkar, P., Jia, R., & Liang, P. (2018). "Know What You Don't Know: Unanswerable Questions for SQuAD." In Proceedings of ACL 2018.  
[3] Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). "Attention Is All You Need." In Advances in Neural Information Processing Systems (NeurIPS 2017).  
[4] Wolf, T., Debut, L., Sanh, V., et al. (2020). "Transformers: State-of-the-Art Natural Language Processing." In Proceedings of EMNLP 2020.  
[5] Streamlit Documentation (2026). "Streamlit API Reference & Caching." https://docs.streamlit.io  

---

## Appendix: Viva Voce Defense Questions & Answers

**Q1: What is the difference between Extractive and Abstractive Question Answering?**  
*Answer:* Extractive QA selects a contiguous substring directly from the source text by predicting start and end token indices, guaranteeing factual accuracy. Abstractive QA generates a novel paraphrased response using a sequence-to-sequence model (like T5 or GPT), which introduces risks of hallucination.

**Q2: What was the main motivation for creating SQuAD 2.0?**  
*Answer:* SQuAD 1.1 only contained answerable questions, causing models to blindly select whatever text had the highest word overlap even when questions made no sense. SQuAD 2.0 introduced over 50,000 adversarial unanswerable questions to train models to know when evidence is missing.

**Q3: How does the model mathematically identify an unanswerable question?**  
*Answer:* During fine-tuning, unanswerable questions are trained with start and end indices pointing to the special token `[CLS]` (index 0). At inference time, the model computes the null score $S_{\text{null}} = s_0 + e_0$. If the best passage span score fails to exceed $S_{\text{null}}$ by threshold $\tau$, the model flags the question as unanswerable.

**Q4: How are subword tokens mapped back to the original passage characters?**  
*Answer:* WordPiece tokenization can split words into fragments (e.g., "playing" into "play" and "##ing"). The fast tokenizer generates an `offset_mapping` array storing the original `(start_char, end_char)` offsets in the raw string. We slice the original string directly using these offsets to preserve whitespace and casing verbatim.

**Q5: Why is `@st.cache_resource` critical for Streamlit deployment?**  
*Answer:* By default, Streamlit re-executes the entire Python script on every user interaction or button click. Loading BERT’s 110M weights into GPU memory takes 2–3 seconds. `@st.cache_resource` caches the loaded model in memory, allowing subsequent clicks to run inference instantly in ~30 ms.
