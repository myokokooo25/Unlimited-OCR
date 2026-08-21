"""CPU tests for infer_local.py helpers. No GPU or model weights required."""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import infer_local


def _write_pdf(path: str, pages: int = 2) -> str:
    import fitz

    doc = fitz.open()
    for i in range(pages):
        page = doc.new_page(width=200, height=200)
        page.insert_text((20, 80), f"Page {i + 1}")
    doc.save(path)
    doc.close()
    return path


class TestCollectImages(unittest.TestCase):
    def test_sorts_and_limits(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "b.png"), "wb") as handle:
                handle.write(b"b")
            with open(os.path.join(tmp, "a.jpg"), "wb") as handle:
                handle.write(b"a")
            with open(os.path.join(tmp, "skip.txt"), "wb") as handle:
                handle.write(b"x")
            names = [os.path.basename(p) for p in infer_local.collect_images(tmp)]
            self.assertEqual(names, ["a.jpg", "b.png"])
            self.assertEqual(len(infer_local.collect_images(tmp, max_images=1)), 1)


class TestPdfToImages(unittest.TestCase):
    def test_max_pages(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = _write_pdf(os.path.join(tmp, "doc.pdf"), pages=3)
            images = infer_local.pdf_to_images(pdf_path, dpi=72, max_pages=2)
            self.assertEqual(len(images), 2)
            self.assertTrue(images[0].endswith("page_0001.png"))


class TestParseArgs(unittest.TestCase):
    def test_pdf_and_max_pages(self):
        args = infer_local.parse_args(["--pdf", "doc.pdf", "--max_pages", "1"])
        self.assertEqual(args.pdf, "doc.pdf")
        self.assertEqual(args.max_pages, 1)
        self.assertFalse(args.check_gpu)

    def test_check_gpu(self):
        args = infer_local.parse_args(["--check-gpu"])
        self.assertTrue(args.check_gpu)


class TestRequireCuda(unittest.TestCase):
    def test_missing_torch(self):
        with patch.dict(sys.modules, {"torch": None}):
            with self.assertRaises(SystemExit):
                infer_local.require_cuda()

    def test_cuda_unavailable(self):
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = False
        with patch.dict(sys.modules, {"torch": fake_torch}):
            with self.assertRaises(SystemExit) as raised:
                infer_local.require_cuda()
        self.assertIn("No NVIDIA CUDA GPU", str(raised.exception))


class TestMainCheckGpu(unittest.TestCase):
    def test_check_gpu_skips_model_load(self):
        with patch("infer_local.require_cuda") as require, patch("infer_local.load_model") as load:
            infer_local.main(["--check-gpu"])
        require.assert_called_once()
        load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
