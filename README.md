# Extractive Question Answering System (SQuAD 2.0 & BERT)

**Project 12 — College NLP Capstone Project**  
**Group:** Group 10  
**Specialisation:** BCA (Artificial Intelligence and Data Science)  
**Repository:** [https://github.com/akash77007/NLP_Question-Answering-System](https://github.com/akash77007/NLP_Question-Answering-System)  

### Team Members (Group 10):
- `2401201113`
- `2401201115`
- `2401201102`
- `2401201213`
- `2401201104`
- `2401201108` (Akash Sharma)

---

## 1. Project Overview & Problem Statement

This project implements an **extractive Question Answering (QA) system** that reads a user-provided context passage and answers questions strictly based on the information present in that passage.

Unlike open-domain generative chatbots that are prone to hallucinating facts or pulling answers from outside memory, this system is **strictly grounded in the provided document**:
1. It identifies the exact **start and end character span** within the passage.
2. It actively rejects **unanswerable questions** where no supporting evidence exists (using SQuAD 2.0 adversarial null-thresholding).
3. It features an interactive, production-ready **Streamlit web GUI** that visually highlights the answer within the source document.

---

## 2. System Architecture

```
[User Question]  +  [Source Passage]
         │
         ▼
[WordPiece BERT Tokenizer]
   - Formatting: [CLS] Question [SEP] Passage [SEP]
   - Generates: input_ids, attention_mask, token_type_ids
   - Computes: character-to-token offset_mapping
         │
         ▼
[BERT-base Transformer Encoder (110M Params)]
   - 12 Bidirectional Transformer layers
   - 768-dimensional contextual hidden representations
         │
         ▼
[QA Span Classification Head]
   - Linear projection: ℝ^768 ➔ ℝ^2
   - Produces per-token Start Logits (s_i) and End Logits (e_j)
         │
         ▼
[SQuAD 2.0 Span Selection & Null Thresholding]
   - Best valid span: S_(i,j) = s_i + e_j  (where i ≤ j, tokens ∈ passage)
   - Null (unanswerable) score: S_null = s_0 + e_0  (at [CLS])
   - Decision Rule:
       If (max(S_(i,j)) - S_null) < Threshold ➔ "No answer found in passage"
       Else ➔ Extract character span [offset[i].start : offset[j].end]
         │
         ▼
[Interactive Streamlit UI]
   - Displays extracted text, confidence %, logit margin, latency
   - Renders live in-context HTML highlight badge
```

---

## 3. Technology Stack & Hardware Setup

- **Language**: Python 3.10+
- **Deep Learning Framework**: PyTorch with CUDA 13.x acceleration
- **Transformer Library**: Hugging Face `transformers`
- **Model Checkpoint**: `deepset/bert-base-cased-squad2`
- **Dataset**: Stanford SQuAD 2.0 (130k training pairs, 11.8k dev pairs)
- **GUI Framework**: Streamlit
- **EDA & Analytics**: Pandas, Matplotlib, OpenPyXL
- **Hardware Acceleration**:
  - GPU: NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM)
  - Processor: AMD Ryzen 7 7435HS
  - OS: Windows 11 64-bit

---

## 4. Repository Structure

```text
Question_Answering_System/
│
├── data/                                      # SQuAD 2.0 datasets
│   ├── squad_2.0_dev_complete_11873.xlsx      # Dev / Validation split
│   ├── squad_2.0_train_complete_130319.xlsx   # Training split
│   └── eval_results.json                      # Automated evaluation metrics output
│
├── notebooks/                                 # Exploratory Data Analysis
│   ├── 01_squad_eda.ipynb                     # Interactive EDA notebook
│   ├── generate_eda_plots.py                  # Script to render EDA figures
│   └── squad_eda_plots.png                    # 4-panel EDA visualization chart
│
├── src/                                       # Core NLP backend engine
│   ├── __init__.py
│   ├── inference.py                           # QuestionAnsweringEngine class
│   └── evaluate.py                            # Exact Match (EM) & Token F1 evaluator
│
├── app.py                                     # Interactive Streamlit Web GUI
├── requirements.txt                           # Frozen project dependencies
└── README.md                                  # Complete capstone documentation
```

---

## 5. Quickstart & Execution Guide

### Step 1: Activate Virtual Environment
```powershell
.\venv\Scripts\Activate.ps1
```

### Step 2: Install Dependencies (if not already installed)
```powershell
pip install -r requirements.txt
```

### Step 3: Run the Streamlit Web Application
```powershell
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### Step 4: Run the SQuAD 2.0 Evaluation Benchmark
```powershell
python -m src.evaluate
```

### Step 5: Generate EDA Charts for Presentation
```powershell
python notebooks/generate_eda_plots.py
```

---

## 6. Mathematical Methodology

### 6.1 Tokenization and Input Representation
For a question $Q = (q_1, \dots, q_m)$ and context passage $C = (c_1, \dots, c_n)$, the sequence is packed as:
$$\mathbf{X} = [\text{CLS}], q_1, \dots, q_m, [\text{SEP}], c_1, \dots, c_n, [\text{SEP}]$$

Segment embeddings (token type IDs) assign $0$ to the question tokens and $1$ to the context tokens.

### 6.2 Logit Computation
BERT outputs contextual vectors $\mathbf{h}_k \in \mathbb{R}^{768}$ for each token index $k$. Two trainable parameter vectors $\mathbf{w}_s, \mathbf{w}_e \in \mathbb{R}^{768}$ project each token:
$$s_k = \mathbf{w}_s^\top \mathbf{h}_k \quad (\text{Start Logit})$$
$$e_k = \mathbf{w}_e^\top \mathbf{h}_k \quad (\text{End Logit})$$

### 6.3 Answer Span Selection
The model searches for start index $i$ and end index $j$ that maximize:
$$S_{(i, j)} = s_i + e_j \quad \text{subject to } i \le j, \; j - i + 1 \le L_{\max}, \; \text{and } \text{token}_i, \text{token}_j \in C$$

### 6.4 Unanswerable (Null) Rejection
In SQuAD 2.0, unanswerable questions have the answer span at the $[\text{CLS}]$ token (index 0).
$$S_{\text{null}} = s_0 + e_0$$
The answer is rejected as unanswerable if:
$$\max_{(i, j)} S_{(i, j)} - S_{\text{null}} < \tau$$
where $\tau$ is the null threshold (calibrated around $-1.0$ to $0.0$).

---

## 7. Model Evaluation Metrics

Performance is benchmarked using standard Stanford SQuAD 2.0 metrics:

1. **Exact Match (EM)**: Binary percentage metric indicating whether the predicted text matches the ground truth verbatim (after lowercasing, punctuation stripping, and article removal).
2. **Token-level F1 Score**: The harmonic mean of token precision and recall:
   $$\text{Precision} = \frac{|\text{Tokens}_{\text{pred}} \cap \text{Tokens}_{\text{gold}}|}{|\text{Tokens}_{\text{pred}}|}, \quad \text{Recall} = \frac{|\text{Tokens}_{\text{pred}} \cap \text{Tokens}_{\text{gold}}|}{|\text{Tokens}_{\text{gold}}|}$$
   $$\text{F1} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
3. **Unanswerable Scoring**: If a question is unanswerable (`is_impossible = True`), the prediction receives $\text{EM} = 1.0, \text{F1} = 1.0$ if it correctly outputs no answer, and $0.0$ otherwise.

---

