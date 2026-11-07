"""Cheap checks for preprocessing and artifact generation; no model/GPU needed."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.data.preprocessing import clean_records, format_chat_prompt, tokenize_example


class FakeTokenizer:
    chat_template = None
    model_max_length = 32

    def apply_chat_template(self, *args, **kwargs):
        raise AssertionError("Fallback format should be used")

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None):
        ids = [ord(char) % 31 for char in text]
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


class SmokeTests(unittest.TestCase):
    def test_clean_records_drops_malformed_and_formats_context(self):
        rows = [
            {"instruction": "  Tell me  ", "context": " context ", "response": " answer "},
            {"instruction": "", "response": "missing prompt"},
            {"instruction": 3, "response": "bad type"},
        ]
        cleaned = clean_records(rows)
        self.assertEqual(cleaned, [{"instruction": "Tell me\n\nContext:\ncontext", "response": "answer"}])

    def test_tokenization_masks_prompt(self):
        tokenizer = FakeTokenizer()
        example = {"instruction": "question", "response": "answer"}
        self.assertEqual(format_chat_prompt(tokenizer, "question"), "User: question\nAssistant:")
        encoded = tokenize_example(example, tokenizer, max_length=64)
        self.assertEqual(len(encoded["input_ids"]), len(encoded["labels"]))
        self.assertIn(-100, encoded["labels"])
        self.assertTrue(any(label != -100 for label in encoded["labels"]))

    def test_full_finetuning_enables_all_parameters(self):
        from src.training.full_finetuning import prepare_full_finetuning

        class Parameter:
            requires_grad = False

        class Model:
            is_loaded_in_4bit = False
            is_loaded_in_8bit = False

            def __init__(self):
                self.values = [Parameter(), Parameter()]

            def parameters(self):
                return self.values

            def gradient_checkpointing_enable(self):
                self.checkpointing = True

            def enable_input_require_grads(self):
                self.input_grads = True

        model = Model()
        prepared = prepare_full_finetuning(model)
        self.assertTrue(all(parameter.requires_grad for parameter in prepared.parameters()))
        self.assertTrue(prepared.checkpointing)
        self.assertTrue(prepared.input_grads)

    def test_pending_report_and_result_serialization(self):
        from src.benchmarking.benchmark import save_results
        from scripts.generate_report import build_report

        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            metrics = root / "metrics"
            save_results([], metrics)
            self.assertEqual(json.loads((metrics / "benchmark_results.json").read_text()), [])
            self.assertIn("pending", build_report([], root / "plots").lower())


if __name__ == "__main__":
    unittest.main()
