import unittest
from src.segmentation import segment_pdf

class TestSegmentation(unittest.TestCase):
    def test_segment_pdf(self):
        toc_structure = {
            "Introduction": "1",
            "Policies and Practices": {
                "Open Door Policy": "3",
                "Employment at Will": "3",
            },
        }
        segments = segment_pdf("tests/sample_document.pdf", toc_structure)
        self.assertIn("Introduction", segments)
        self.assertIn("Open Door Policy", segments)
        self.assertNotIn("Nonexistent Section", segments)

if __name__ == "__main__":
    unittest.main()
