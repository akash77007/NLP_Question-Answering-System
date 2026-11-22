"""
tests/test_qa.py
Unit and edge-case testing for the Extractive QA Engine.
"""

import unittest
from src.inference import QuestionAnsweringEngine


class TestQuestionAnsweringEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n[Setup] Loading QA Engine for unit testing...")
        cls.engine = QuestionAnsweringEngine()

        cls.sample_context = (
            "The solar system consists of the Sun and the objects that orbit it. "
            "Of the planets, Mercury is the closest to the Sun, while Neptune is the farthest. "
            "Jupiter is the largest planet in our solar system by both mass and volume."
        )

    def test_answerable_question(self):
        """Test standard answer extraction."""
        question = "Which planet is the closest to the Sun?"
        res = self.engine.predict(question, self.sample_context)
        self.assertFalse(res["is_impossible"])
        self.assertIn("Mercury", res["answer"])
        self.assertGreater(res["confidence"], 0.70)
        self.assertGreater(res["latency_ms"], 0)

    def test_unanswerable_question(self):
        """Test rejection of question not present in context."""
        question = "What is the speed of light in kilometers per second?"
        res = self.engine.predict(question, self.sample_context)
        self.assertTrue(res["is_impossible"])
        self.assertEqual(res["answer"], "No answer found in the provided passage.")

    def test_empty_inputs(self):
        """Test handling of empty passage or question."""
        res1 = self.engine.predict("", self.sample_context)
        self.assertTrue(res1["is_impossible"])

        res2 = self.engine.predict("Valid question?", "")
        self.assertTrue(res2["is_impossible"])

    def test_long_passage(self):
        """Test robustness on long context passages exceeding standard 384-token windows."""
        long_context = self.sample_context * 10
        question = "What is the largest planet?"
        res = self.engine.predict(question, long_context)
        self.assertFalse(res["is_impossible"])
        self.assertIn("Jupiter", res["answer"])


if __name__ == "__main__":
    unittest.main()
