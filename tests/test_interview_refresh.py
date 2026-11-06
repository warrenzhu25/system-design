from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
REFRESH_HEADING = "## Additions — November 2026 Refresh"
# file -> number of python blocks expected in the refresh section
EXPECTED_BLOCKS = {
    "openai_interview_questions.md": 6,
    "anthropic_interview_questions.md": 6,
    "airbnb_interview_questions.md": 6,
}


def refresh_section(text):
    start = text.index(REFRESH_HEADING)
    end = text.find("\n## ", start + len(REFRESH_HEADING))
    return text[start:end if end != -1 else len(text)]


def python_blocks(text):
    return re.findall(r"^```python\s*\n(.*?)^```\s*$", text, re.MULTILINE | re.DOTALL)


class InterviewRefreshTest(unittest.TestCase):
    def test_refresh_sections_run_and_pass_their_asserts(self):
        for name, expected in EXPECTED_BLOCKS.items():
            blocks = python_blocks(refresh_section((ROOT / name).read_text(encoding="utf-8")))
            with self.subTest(file=name):
                self.assertEqual(len(blocks), expected)
                for index, block in enumerate(blocks, start=1):
                    self.assertIn("assert ", block)
                    code = compile(block, f"<{name} block {index}>", "exec")
                    exec(code, {"__name__": f"refresh_{index}"})

    def test_linkedin_guide_blocks_run_and_pass_their_asserts(self):
        blocks = python_blocks((ROOT / "linkedin_interview_questions.md").read_text(encoding="utf-8"))
        self.assertEqual(len(blocks), 12)
        for index, block in enumerate(blocks, start=1):
            with self.subTest(block=index):
                self.assertIn("assert ", block)
                code = compile(block, f"<linkedin block {index}>", "exec")
                exec(code, {"__name__": f"linkedin_{index}"})


if __name__ == "__main__":
    unittest.main()
