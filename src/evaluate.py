"""
src/evaluate.py
Standard SQuAD 2.0 evaluation metrics: Exact Match (EM) and Token-level F1.
Includes text normalization, multi-reference ground-truth matching,
and evaluation breakdown for Answerable (HasAns) and Unanswerable (NoAns) questions.
"""

import os
import re
import string
import json
import ast
import time
from collections import Counter
from typing import List, Dict, Any, Tuple
import pandas as pd
from tqdm import tqdm
from .inference import QuestionAnsweringEngine


def normalize_answer(text: str) -> str:
    """
    Standard SQuAD normalization:
    - Lowercase text
    - Remove punctuation
    - Remove English articles ('a', 'an', 'the')
    - Standardize whitespace
    """
    text = str(text).lower()

    # Remove punctuation
    text = "".join(ch for ch in text if ch not in set(string.punctuation))

    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)

    # Clean whitespace
    text = " ".join(text.split())
    return text


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """Computes binary exact match score (1.0 or 0.0) between normalized texts."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def compute_f1(prediction: str, ground_truth: str) -> float:
    """
    Computes token-level F1 score between prediction and ground truth.
    Harmonic mean of precision and recall over bag-of-words tokens.
    """
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(gold_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def evaluate_sample(
    prediction: str,
    is_pred_impossible: bool,
    gold_answers: List[str],
    is_gold_impossible: bool
) -> Tuple[float, float]:
    """
    Evaluates a single prediction against ground-truth according to SQuAD 2.0 rules.

    Returns:
        (exact_match, f1)
    """
    # Case 1: Ground truth is Unanswerable
    if is_gold_impossible or not gold_answers:
        if is_pred_impossible or normalize_answer(prediction) == "":
            return (1.0, 1.0)
        else:
            return (0.0, 0.0)

    # Case 2: Ground truth is Answerable
    if is_pred_impossible or normalize_answer(prediction) == "":
        return (0.0, 0.0)

    # Take maximum EM and F1 across all human reference answers
    em = max(compute_exact_match(prediction, gold) for gold in gold_answers)
    f1 = max(compute_f1(prediction, gold) for gold in gold_answers)
    return (em, f1)


def parse_cell(val: Any) -> Any:
    """Parse string representations of Python lists from Excel/CSV."""
    if isinstance(val, list):
        return val
    if pd.isna(val) or val is None or val == "--" or str(val).strip() == "":
        return []
    s = str(val).strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            return ast.literal_eval(s)
        except Exception:
            return [s[1:-1].strip("'\"")]
    return [s]


def run_evaluation(
    data_path: str,
    num_samples: int = 200,
    null_threshold: float = -1.0,
    output_json: str = "data/eval_results.json"
) -> Dict[str, Any]:
    """
    Runs model evaluation across a representative sample of SQuAD 2.0.
    """
    print(f"\n[Evaluation] Loading evaluation dataset from: {data_path}")
    if data_path.endswith(".xlsx"):
        df = pd.read_excel(data_path)
    else:
        df = pd.read_csv(data_path)

    print(f"[Evaluation] Total available rows: {len(df)}")

    # Sample balanced dataset (half answerable, half unanswerable)
    df_ans = df[df["is_impossible"] == False]
    df_unans = df[df["is_impossible"] == True]

    n_ans = min(num_samples // 2, len(df_ans))
    n_unans = min(num_samples - n_ans, len(df_unans))

    eval_df = pd.concat([
        df_ans.sample(n=n_ans, random_state=42),
        df_unans.sample(n=n_unans, random_state=42)
    ]).sample(frac=1.0, random_state=42).reset_index(drop=True)

    print(f"[Evaluation] Evaluating on {len(eval_df)} balanced samples ({n_ans} Answerable, {n_unans} Unanswerable)...")

    qa_engine = QuestionAnsweringEngine()

    total_em = 0.0
    total_f1 = 0.0

    has_ans_em = 0.0
    has_ans_f1 = 0.0
    has_ans_count = 0

    no_ans_em = 0.0
    no_ans_f1 = 0.0
    no_ans_count = 0

    latencies = []
    detailed_results = []

    for idx, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc="Evaluating QA Model"):
        # Context extraction and cleanup
        raw_ctx = str(row["context"])
        if raw_ctx.startswith("['") and raw_ctx.endswith("']"):
            context = raw_ctx[2:-2]
        else:
            context = raw_ctx

        question = str(row["question"])
        is_impossible = bool(row["is_impossible"])
        gold_answers = parse_cell(row["answers"])

        # Predict
        res = qa_engine.predict(question, context, null_threshold=null_threshold)
        latencies.append(res["latency_ms"])

        pred_answer = "" if res["is_impossible"] else res["answer"]
        em, f1 = evaluate_sample(
            pred_answer,
            res["is_impossible"],
            gold_answers,
            is_impossible
        )

        total_em += em
        total_f1 += f1

        if is_impossible:
            no_ans_em += em
            no_ans_f1 += f1
            no_ans_count += 1
        else:
            has_ans_em += em
            has_ans_f1 += f1
            has_ans_count += 1

        detailed_results.append({
            "question": question,
            "prediction": res["answer"],
            "pred_is_impossible": res["is_impossible"],
            "gold_answers": gold_answers,
            "gold_is_impossible": is_impossible,
            "em": em,
            "f1": round(f1, 4),
            "latency_ms": res["latency_ms"]
        })

    n = len(eval_df)
    results = {
        "model_name": qa_engine.model_name,
        "total_samples": n,
        "exact_match": round((total_em / n) * 100, 2),
        "f1_score": round((total_f1 / n) * 100, 2),
        "has_answer_exact_match": round((has_ans_em / max(1, has_ans_count)) * 100, 2),
        "has_answer_f1": round((has_ans_f1 / max(1, has_ans_count)) * 100, 2),
        "has_answer_count": has_ans_count,
        "no_answer_exact_match": round((no_ans_em / max(1, no_ans_count)) * 100, 2),
        "no_answer_f1": round((no_ans_f1 / max(1, no_ans_count)) * 100, 2),
        "no_answer_count": no_ans_count,
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2),
        "null_threshold": null_threshold
    }

    # Print summary table
    print("\n" + "=" * 55)
    print("               SQuAD 2.0 EVALUATION RESULTS             ")
    print("=" * 55)
    print(f"Model Checkpoint:       {results['model_name']}")
    print(f"Total Samples Evaluated: {results['total_samples']}")
    print(f"Average Inference Time:  {results['avg_latency_ms']} ms / query")
    print("-" * 55)
    print(f"Overall Exact Match (EM): {results['exact_match']}%")
    print(f"Overall F1 Score:         {results['f1_score']}%")
    print("-" * 55)
    print(f"Has-Answer EM (Answerable):   {results['has_answer_exact_match']}%  (n={has_ans_count})")
    print(f"Has-Answer F1 (Answerable):   {results['has_answer_f1']}%")
    print("-" * 55)
    print(f"No-Answer EM (Unanswerable):  {results['no_answer_exact_match']}%  (n={no_ans_count})")
    print(f"No-Answer F1 (Unanswerable):  {results['no_answer_f1']}%")
    print("=" * 55 + "\n")

    # Save to disk
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump({"summary": results, "details": detailed_results[:20]}, f, indent=2)
    print(f"[+] Evaluation results saved to: {output_json}")

    return results


if __name__ == "__main__":
    dev_path = os.path.join("data", "squad_2.0_dev_complete_11873.xlsx")
    if not os.path.exists(dev_path):
        dev_path = os.path.join("..", "data", "squad_2.0_dev_complete_11873.xlsx")
    run_evaluation(dev_path, num_samples=100)
