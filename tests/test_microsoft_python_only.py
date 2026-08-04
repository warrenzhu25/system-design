from pathlib import Path
import re
import unittest


DOCUMENT = Path(__file__).resolve().parents[1] / "microsoft_interview_questions.md"


class MicrosoftPythonOnlyTest(unittest.TestCase):
    def test_document_has_only_python_solution_sections(self):
        text = DOCUMENT.read_text(encoding="utf-8")

        self.assertIn("Every coding solution below is implemented and tested in Python.", text)
        self.assertNotRegex(text, r"\*\*(?:Java|TypeScript) Solution:\*\*")
        self.assertNotRegex(text, r"^```(?:java|typescript|lua)\s*$", re.MULTILINE)
        self.assertEqual(len(re.findall(r"\*\*Python Solution:\*\*", text)), 5)
        self.assertIn("**Reference implementation of the algorithmic core**", text)


if __name__ == "__main__":
    unittest.main()
