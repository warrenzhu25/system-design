from pathlib import Path
import re
import unittest


DOCUMENT = Path(__file__).resolve().parents[1] / "datapelago_coding_interview_python.md"
EXPECTED_CODING_QUESTIONS = 13


def python_blocks(text):
    return re.findall(r"^```python\s*\n(.*?)^```\s*$", text, re.MULTILINE | re.DOTALL)


class DataPelagoCodingGuideTest(unittest.TestCase):
    def test_document_declares_scope_and_provenance(self):
        text = DOCUMENT.read_text(encoding="utf-8")

        self.assertIn("Every coding solution below is implemented and tested in Python.", text)
        self.assertIn("not published by DataPelago", text)
        for tag in ("[General]", "[Source]", "[Ask]"):
            self.assertIn(tag, text)

    def test_every_coding_question_has_a_python_solution(self):
        text = DOCUMENT.read_text(encoding="utf-8")

        headings = re.findall(r"^### Q\d+\.", text, re.MULTILINE)
        self.assertEqual(len(headings), EXPECTED_CODING_QUESTIONS)
        self.assertEqual(len(re.findall(r"\*\*Python Solution:\*\*", text)), EXPECTED_CODING_QUESTIONS)
        self.assertEqual(len(python_blocks(text)), EXPECTED_CODING_QUESTIONS)

    def test_every_python_fence_runs_and_passes_its_own_asserts(self):
        text = DOCUMENT.read_text(encoding="utf-8")

        for index, block in enumerate(python_blocks(text), start=1):
            with self.subTest(block=index):
                self.assertIn("assert ", block)
                code = compile(block, f"<DataPelago coding block {index}>", "exec")
                exec(code, {"__name__": f"datapelago_block_{index}"})


if __name__ == "__main__":
    unittest.main()
