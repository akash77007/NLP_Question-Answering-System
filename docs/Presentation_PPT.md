# PowerPoint Presentation Deck (10 Slides)
## Project 12: Extractive Question Answering System
**Specialisation:** BCA (Artificial Intelligence & Data Science)

---

### Slide 1: Title & Team Details
- **Project Title:** DocuBERT: Context-Grounded Extractive Question Answering System
- **Project Assignment:** Project 12 — Question Answering System
- **Group:** Group 10
- **Domain:** Natural Language Processing & Deep Learning
- **Technology Stack:** PyTorch, BERT-base, Hugging Face, SQuAD 2.0, Streamlit
- **Hardware Acceleration:** NVIDIA GeForce RTX 3050 Laptop GPU (CUDA 13.x)
- **GitHub Repository:** [https://github.com/akash77007/NLP_Question-Answering-System](https://github.com/akash77007/NLP_Question-Answering-System)
- **Team Members (Group 10):**
  - Roll No: 2401201113
  - Roll No: 2401201115
  - Roll No: 2401201102
  - Roll No: 2401201213
  - Roll No: 2401201104
  - Roll No: 2401201108 (Akash Sharma)
- **Course & Department:** BCA (Artificial Intelligence & Data Science)

> **Speaker Note:** *"Good morning respected evaluators. Today we present Project 12: An Extractive Question Answering System built using BERT and Stanford SQuAD 2.0, fully accelerated on an NVIDIA RTX 3050 GPU."*

---

### Slide 2: Problem Statement & Objectives
- **The Core Problem:**
  - Modern generative LLMs frequently hallucinate facts or bring in unverified external training data.
  - In critical domains (legal, medical, enterprise documents), users need answers **strictly extracted** from the provided text.
- **Key Objectives:**
  - Build an extractive reading comprehension model that predicts the exact answer span from a user-provided passage.
  - Implement unanswerable query detection so the system rejects questions with no evidence in the text.
  - Eliminate hallucination with strict factual grounding.
  - Deploy a responsive, user-friendly Streamlit web GUI with real-time in-passage visual highlighting.

> **Speaker Note:** *"Unlike open-ended chatbots that guess or hallucinate, our extractive system identifies the exact start and end boundaries inside the document. If the answer does not exist, it explicitly alerts the user."*

---

### Slide 3: Introduction & Real-World Applications
- **What is Extractive QA?**
  - An NLP task where a model reads a document and answers questions by extracting contiguous spans of text rather than synthesizing new sentences.
- **Industry Applications:**
  - **Legal Document Review:** Locating indemnification clauses and liability terms within lengthy contracts.
  - **Healthcare & Clinical Guidelines:** Pulling dosage and contraindication spans from medical literature.
  - **Financial Due Diligence:** Extracting EBITDA and risk disclosures from annual 10-K filings.
  - **Enterprise Search:** Powering customer support knowledge bases with verbatim documentation links.

> **Speaker Note:** *"Extractive QA is the foundation of high-precision enterprise document search. It guarantees that every single answer can be audited against the original text."*

---

### Slide 4: Dataset Description — Stanford SQuAD 2.0
- **Dataset Choice:** Stanford Question Answering Dataset 2.0 (SQuAD 2.0).
- **Scale:**
  - Training Set: 130,319 question-passage pairs.
  - Development / Validation Set: 11,873 question-passage pairs.
- **Why SQuAD 2.0 instead of SQuAD 1.1?**
  - SQuAD 1.1 assumed every question had an answer, forcing models to guess.
  - SQuAD 2.0 introduced over 50,000 **adversarial, unanswerable questions** written by crowdworkers to look superficially relevant.
  - SQuAD 2.0 forces the model to learn when *not* to answer.

> **Speaker Note:** *"We specifically used SQuAD 2.0 because over 50% of the benchmark questions are unanswerable. This prevents our model from giving false positives on trick questions."*

---

### Slide 5: Data Preprocessing & Exploratory Data Analysis (EDA)
- **Key EDA Findings (from `data/squad_eda_plots.png`):**
  - **Class Distribution:** 49.93% Answerable vs. 50.07% Unanswerable (perfectly balanced validation set).
  - **Passage Length:** Median context is 120 words; 95th percentile is 220 words. (Fits within BERT's 384-token window).
  - **Question Length:** Median is 10 words; 95th percentile is 15 words.
  - **Linguistic Breakdown:** Top interrogative words: *What* (58%), *Who* (14%), *How* (10%), *When* (9%), *Where* (5%).
- **Preprocessing Pipeline:**
  - Subword WordPiece tokenization (30,522 vocabulary size).
  - Joint pair format: `[CLS] Question [SEP] Context [SEP]`.
  - Preservation of character offset mappings for lossless detokenization.

> **Speaker Note:** *"Our EDA revealed that 95% of SQuAD passages are under 220 words. Because BERT tokenizes words into roughly 1.3 subwords, a max sequence length of 384 comfortably covers the vast majority without truncation."*

---

### Slide 6: Feature Engineering & Methodology
- **Bidirectional Self-Attention:**
  - Unlike directional RNNs, BERT attends to left and right context simultaneously across all 12 Transformer layers.
  - Attention formula:
    $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$
- **Triple Input Embeddings:**
  1. **Token Embeddings:** Subword representation.
  2. **Segment Embeddings:** Binary vector (0 for question, 1 for passage).
  3. **Position Embeddings:** Positional vectors encoding absolute token indices (0 to 511).

> **Speaker Note:** *"The segment embeddings are vital because they inform the attention heads which tokens belong to the user's question and which belong to the source passage."*

---

### Slide 7: Model Architecture & Algorithms
- **Model Checkpoint:** `deepset/bert-base-cased-squad2` (110M parameters).
- **Extractive Classification Head:**
  - A dense linear layer projects each token's hidden state $h_i \in \mathbb{R}^{768}$ into two logits:
    - $s_i = \mathbf{w}_s^\top h_i$ (Start Logit)
    - $e_j = \mathbf{w}_e^\top h_j$ (End Logit)
- **Span Extraction Algorithm:**
  - Maximize span score: $S_{(i, j)} = s_i + e_j$ subject to $i \le j$ and tokens $\in \text{Passage}$.
- **Unanswerable Detection:**
  - Null score at `[CLS]`: $S_{\text{null}} = s_0 + e_0$.
  - Decision Rule: If $\max(S_{(i, j)}) - S_{\text{null}} < \tau$, flag as unanswerable.

> **Speaker Note:** *"The `[CLS]` token acts as the designated null token. During training on SQuAD 2.0, if an answer is impossible, the ground truth start and end indices point to index 0."*

---

### Slide 8: Performance Evaluation & Results
- **Evaluation Metrics (Tested on Local SQuAD 2.0 Benchmark):**
  - **Overall Exact Match (EM):** **64.00%**
  - **Overall Token F1 Score:** **69.23%**
  - **Unanswerable (No-Answer) F1:** **74.00%**
  - **Answerable (Has-Answer) F1:** **64.46%**
  - **Hardware Latency:** **32.85 ms / query** on NVIDIA RTX 3050 GPU (~30 queries/sec).
- **Error Analysis:**
  - Partial boundary mismatches (e.g., predicting *"Court of Justice"* instead of *"the Court of Justice of the European Union"* gets 0% EM but ~92% F1).

> **Speaker Note:** *"Our model achieved an overall F1 of 69.23% and correctly rejected unanswerable questions 74% of the time, operating at an average latency of just 32 milliseconds per query."*

---

### Slide 9: Working GUI & Demonstration
- **Streamlit Application Features:**
  - **Document Input Area:** Paste any custom text or select from curated presets (Science, History, Technology).
  - **Adversarial Test Preset:** Live trick questions to demonstrate no-answer rejection.
  - **In-Passage Visual Highlighting:** Uses HTML span badges to visually illuminate the exact answer location.
  - **Model Metrics Display:** Real-time feedback showing Model Confidence %, Logit Margin ($\Delta$), and Latency in milliseconds.
  - **Threshold Slider:** Sidebar control to adjust model sensitivity dynamically.
  - **GPU Memory Caching:** Implements `@st.cache_resource` for zero re-load delay.

> **Speaker Note:** *"Now we will demonstrate our working Streamlit web interface live, testing both answerable questions and adversarial trick questions."*

---

### Slide 10: Conclusion & Future Scope
- **Conclusion:**
  - Successfully built and validated an extractive QA system grounded strictly in source passages.
  - Effectively mitigated hallucinations using SQuAD 2.0 null-thresholding.
  - Achieved high throughput (~30 ms/query) on local laptop GPU hardware.
- **Future Scope:**
  - **Multi-document Retrieval (RAG):** Adding BM25 or dense vector retrieval (FAISS) to scale from single passages to full document libraries.
  - **Model Quantization:** Applying INT8 / ONNX Runtime quantization to reduce VRAM footprint for mobile deployment.
  - **Multilingual Support:** Transitioning to `XLM-RoBERTa` for multilingual extractive comprehension across Indic languages.

> **Speaker Note:** *"Thank you for your time. We are now ready for your questions."*
