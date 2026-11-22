"""
src/inference.py
Production-grade Extractive Question Answering Engine using BERT and SQuAD 2.0.
"""

import time
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from typing import Dict, Any, Optional


class QuestionAnsweringEngine:
    """
    Extractive Question Answering pipeline built on fine-tuned BERT (SQuAD 2.0).
    Handles question-passage tokenization, start/end span extraction,
    unanswerable query rejection via null score ([CLS]), and character offset alignment.
    """

    def __init__(
        self,
        model_name: str = "deepset/bert-base-cased-squad2",
        device: Optional[str] = None
    ):
        self.model_name = model_name
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[QA Engine] Initializing model '{model_name}' on device: {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        print("[QA Engine] Model successfully loaded and ready for inference.")

    def predict(
        self,
        question: str,
        context: str,
        null_threshold: float = -1.0,
        max_answer_length: int = 30,
        max_seq_length: int = 384,
        doc_stride: int = 128
    ) -> Dict[str, Any]:
        """
        Extract the most probable answer span from `context` for the given `question`.

        Args:
            question: The user query string.
            context: The source passage string.
            null_threshold: Threshold difference (best_score - null_score).
                           If (best_score - null_score) < threshold, the question is
                           deemed unanswerable. Lower values make the model more likely
                           to return an answer; higher values make it more cautious.
            max_answer_length: Maximum number of tokens permitted in an answer span.
            max_seq_length: Maximum token length for the combined [CLS] + Q + [SEP] + Context + [SEP].
            doc_stride: Stride step if sliding window were needed.

        Returns:
            Dict containing:
                - answer (str): Extracted text or unanswerable notice
                - is_impossible (bool): True if no valid answer found/supported
                - score (float): Raw score difference (best_span_score - null_score)
                - confidence (float): Calibrated probability score (0.0 to 1.0)
                - start_char (int): Character index where answer begins in context
                - end_char (int): Character index where answer ends in context
                - latency_ms (float): Inference duration in milliseconds
        """
        start_time = time.perf_counter()

        # Input sanitization
        clean_q = str(question).strip()
        clean_c = str(context).strip()

        if not clean_q or not clean_c:
            return {
                "answer": "Please provide both a valid passage and a question.",
                "is_impossible": True,
                "score": 0.0,
                "confidence": 0.0,
                "start_char": 0,
                "end_char": 0,
                "latency_ms": 0.0,
            }

        # Tokenize question and passage together with fast Rust tokenizer
        encoding = self.tokenizer(
            clean_q,
            clean_c,
            max_length=max_seq_length,
            truncation="only_second",  # Truncate context if total exceeds max_seq_length
            padding="max_length",
            return_tensors="pt",
            return_offsets_mapping=True,
            return_overflowing_tokens=False
        )

        offset_mapping = encoding.pop("offset_mapping")[0].tolist()
        inputs = {k: v.to(self.device) for k, v in encoding.items()}

        # Identify which tokens belong to the context passage
        sequence_ids = encoding.sequence_ids(0)

        with torch.no_grad():
            outputs = self.model(**inputs)

        start_logits = outputs.start_logits[0].cpu()
        end_logits = outputs.end_logits[0].cpu()

        # Score of the [CLS] token (null / unanswerable score)
        # In SQuAD 2.0 fine-tuning, [CLS] is trained to be the start and end of unanswerable questions
        cls_index = 0
        score_null = (start_logits[cls_index] + end_logits[cls_index]).item()

        # Find valid passage token indices (sequence_id == 1)
        valid_indices = [
            idx for idx, seq_id in enumerate(sequence_ids)
            if seq_id == 1 and idx < len(offset_mapping)
        ]

        if not valid_indices:
            latency = (time.perf_counter() - start_time) * 1000
            return {
                "answer": "No answer found in the provided passage.",
                "is_impossible": True,
                "score": score_null,
                "confidence": 0.0,
                "start_char": 0,
                "end_char": 0,
                "latency_ms": round(latency, 2),
            }

        # Search top candidate spans within the context tokens
        min_idx = valid_indices[0]
        max_idx = valid_indices[-1]

        best_score = float("-inf")
        best_span = (0, 0)

        # Get top-k start and end indices to constrain search efficiently
        k = min(20, len(valid_indices))
        top_start_indices = torch.topk(start_logits[min_idx : max_idx + 1], k=k).indices + min_idx
        top_end_indices = torch.topk(end_logits[min_idx : max_idx + 1], k=k).indices + min_idx

        for s_idx in top_start_indices.tolist():
            for e_idx in top_end_indices.tolist():
                if e_idx < s_idx:
                    continue
                if (e_idx - s_idx + 1) > max_answer_length:
                    continue

                span_score = (start_logits[s_idx] + end_logits[e_idx]).item()
                if span_score > best_score:
                    best_score = span_score
                    best_span = (s_idx, e_idx)

        # Calculate score difference relative to unanswerable ([CLS])
        score_diff = best_score - score_null
        latency = (time.perf_counter() - start_time) * 1000

        # Calibrated confidence via sigmoid of the score margin
        confidence = float(torch.sigmoid(torch.tensor(score_diff)).item())

        # If best span score is weaker than null score by threshold -> Unanswerable
        if score_diff < null_threshold or best_score == float("-inf"):
            return {
                "answer": "No answer found in the provided passage.",
                "is_impossible": True,
                "score": round(score_diff, 4),
                "confidence": round(1.0 - confidence, 4),
                "start_char": 0,
                "end_char": 0,
                "latency_ms": round(latency, 2),
            }

        # Extract answer text using character offset mappings to preserve original text verbatim
        start_char, _ = offset_mapping[best_span[0]]
        _, end_char = offset_mapping[best_span[1]]

        extracted_text = clean_c[start_char:end_char].strip()

        # Fallback if character offsets point to empty slice
        if not extracted_text:
            extracted_text = self.tokenizer.decode(
                inputs["input_ids"][0][best_span[0] : best_span[1] + 1],
                skip_special_tokens=True
            ).strip()

        return {
            "answer": extracted_text,
            "is_impossible": False,
            "score": round(score_diff, 4),
            "confidence": round(confidence, 4),
            "start_char": start_char,
            "end_char": end_char,
            "latency_ms": round(latency, 2),
        }


if __name__ == "__main__":
    # Self-test when run directly
    qa = QuestionAnsweringEngine()

    sample_context = (
        "The Amazon rainforest, alternatively the Amazon Jungle or Amazonia, is a moist broadleaf tropical "
        "rainforest in the Amazon biome that covers most of the Amazon basin of South America. This basin "
        "encompasses 7,000,000 km2 (2,700,000 sq mi), of which 5,500,000 km2 (2,100,000 sq mi) are covered by the rainforest."
    )

    print("\n--- Test 1: Answerable Query ---")
    q1 = "How much of the Amazon basin is covered by rainforest?"
    res1 = qa.predict(q1, sample_context)
    print(f"Q: {q1}")
    print(f"Ans: {res1['answer']} (Confidence: {res1['confidence']:.2%}, Latency: {res1['latency_ms']}ms)")

    print("\n--- Test 2: Unanswerable / Impossible Query ---")
    q2 = "Who is the president of the Amazon rainforest?"
    res2 = qa.predict(q2, sample_context)
    print(f"Q: {q2}")
    print(f"Ans: {res2['answer']} (Is Impossible: {res2['is_impossible']}, Confidence: {res2['confidence']:.2%})")
