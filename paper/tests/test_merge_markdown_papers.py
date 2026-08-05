import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPOSITORY_ROOT / "scripts" / "merge_markdown_papers.py"


def load_merge_module():
    spec = importlib.util.spec_from_file_location("merge_markdown_papers", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class MergeMarkdownTest(unittest.TestCase):
    def test_merges_markdown_in_filename_order_and_excludes_output(self):
        merger = load_merge_module()

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            (source / "beta.md").write_text("# Beta\n\nBeta text.\n", encoding="utf-8")
            (source / "alpha.md").write_text("# Alpha\n\nAlpha text.\n", encoding="utf-8")
            output = source / "all-papers.md"
            output.write_text("stale output", encoding="utf-8")

            merger.merge_markdown(source, output)
            result = output.read_text(encoding="utf-8")

        self.assertLess(result.index("# Alpha"), result.index("# Beta"))
        self.assertNotIn("stale output", result)
        self.assertIn("- [Alpha](#paper-alpha)", result)
        self.assertIn("- [Beta](#paper-beta)", result)
        self.assertEqual(result.count("## Paper: "), 2)

    def test_uses_default_title_and_subtitle(self):
        merger = load_merge_module()

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            (source / "alpha.md").write_text("# Alpha\n\nAlpha text.\n", encoding="utf-8")
            output = source / "all-papers.md"

            merger.merge_markdown(source, output)
            result = output.read_text(encoding="utf-8")

        self.assertTrue(result.startswith("# Database Systems Papers\n"))
        self.assertIn("> Combined text-only Markdown volume.", result)

    def test_overrides_title_and_subtitle(self):
        merger = load_merge_module()

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            (source / "alpha.md").write_text("# Alpha\n\nAlpha text.\n", encoding="utf-8")
            output = source / "all-notes.md"

            merger.merge_markdown(
                source,
                output,
                title="Reading Notes",
                subtitle="Study aids, not substitutes for the papers.",
            )
            result = output.read_text(encoding="utf-8")

        self.assertTrue(result.startswith("# Reading Notes\n"))
        self.assertIn("> Study aids, not substitutes for the papers.", result)
        self.assertNotIn("Database Systems Papers", result)


if __name__ == "__main__":
    unittest.main()
