import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPOSITORY_ROOT / "scripts" / "convert_pdfs_to_markdown.py"


def load_converter_module():
    spec = importlib.util.spec_from_file_location("convert_pdfs_to_markdown", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class CleanMarkdownTest(unittest.TestCase):
    def test_removes_markdown_and_html_images(self):
        converter = load_converter_module()
        markdown = """# Paper

Before.

![architecture](assets/architecture.png)

<img src="diagram.svg" alt="diagram">

After.
"""

        cleaned = converter.clean_markdown(markdown)

        self.assertEqual(cleaned, "# Paper\n\nBefore.\n\nAfter.\n")

    def test_normalizes_pdf_artifacts_without_losing_text(self):
        converter = load_converter_module()
        markdown = "First\u00a0paragraph.\x00\n\n\n\nSecond paragraph.\n"

        cleaned = converter.clean_markdown(markdown)

        self.assertEqual(cleaned, "First paragraph.\n\nSecond paragraph.\n")


class ConversionTest(unittest.TestCase):
    def test_writes_text_only_markdown_with_source_header(self):
        converter = load_converter_module()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sample-paper.pdf"
            destination = root / "sample-paper.md"
            source.write_bytes(b"not needed by the fake extractor")

            converter.convert_pdf(
                source,
                destination,
                extractor=lambda _: "# Sample Paper\n\nBody text.\n\n![](figure.png)\n",
            )

            result = destination.read_text(encoding="utf-8")

        self.assertIn("Text-only Markdown conversion", result)
        self.assertIn("[source PDF](../pdf/sample-paper.pdf)", result)
        self.assertIn("Body text.", result)
        self.assertNotIn("figure.png", result)

    def test_excludes_text_drawn_inside_vector_figure(self):
        converter = load_converter_module()
        try:
            import fitz
        except ImportError:
            self.skipTest("PyMuPDF is not installed")

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "vector-figure.pdf"
            document = fitz.open()
            page = document.new_page()
            page.draw_rect(fitz.Rect(72, 100, 300, 200))
            page.insert_text((100, 150), "DIAGRAM NODE")
            page.insert_text((72, 220), "Figure 1. Example architecture.")
            page.insert_text((72, 260), "Body text must remain.")
            document.save(source)
            document.close()

            markdown = converter.extract_text_markdown(source)

        self.assertNotIn("DIAGRAM NODE", markdown)
        self.assertIn("Figure 1. Example architecture.", markdown)
        self.assertIn("Body text must remain.", markdown)


if __name__ == "__main__":
    unittest.main()
