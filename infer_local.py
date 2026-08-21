"""Local NVIDIA GPU inference with Hugging Face Transformers.

This is the path to use on a personal Windows/Linux PC with an NVIDIA GPU.
It does not start SGLang and does not need the cloud agent environment.

Usage (after setup_local.ps1 / setup_local.sh):
    python infer_local.py --check-gpu
    python infer_local.py --pdf .\\Unlimited-OCR.pdf --output_dir .\\outputs --max_pages 1
    python infer_local.py --image_dir .\\pages --output_dir .\\outputs
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time

PROMPT_SINGLE = "<image>document parsing."
PROMPT_MULTI = "<image>Multi page parsing."
NO_REPEAT_NGRAM_SIZE = 35
PDF_DPI = 300
IMAGE_MODES = {
    "gundam": {"base_size": 1024, "image_size": 640, "crop_mode": True, "ngram_window": 128},
    "base": {"base_size": 1024, "image_size": 1024, "crop_mode": False, "ngram_window": 1024},
}


def pdf_to_images(pdf_path: str, dpi: int = PDF_DPI, max_pages: int = 0) -> list[str]:
    import fitz

    doc = fitz.open(pdf_path)
    tmp_dir = tempfile.mkdtemp(prefix="pdf_ocr_")
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    image_paths = []
    page_count = doc.page_count
    limit = page_count if max_pages <= 0 else min(page_count, max_pages)
    for i in range(limit):
        out_path = os.path.join(tmp_dir, f"page_{i + 1:04d}.png")
        doc[i].get_pixmap(matrix=mat).save(out_path)
        image_paths.append(out_path)
    doc.close()
    return image_paths


def collect_images(image_dir: str, max_images: int = 0) -> list[str]:
    exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    image_files = []
    for root, _, files in os.walk(image_dir):
        for name in files:
            if name.lower().endswith(exts):
                image_files.append(os.path.join(root, name))
    image_files.sort()
    if max_images > 0:
        return image_files[:max_images]
    return image_files


def require_cuda() -> str:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "PyTorch is not installed. Run setup_local.ps1 (Windows) or setup_local.sh (Linux)."
        ) from exc

    if not torch.cuda.is_available():
        raise SystemExit(
            "No NVIDIA CUDA GPU found.\n"
            "This script is for a local PC with NVIDIA drivers.\n"
            "Check `nvidia-smi` in PowerShell/CMD, then reinstall torch with CUDA."
        )
    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / (1024 ** 3)
    print(f"GPU: {torch.cuda.get_device_name(0)}  VRAM: {vram_gb:.1f} GB  CUDA: {torch.version.cuda}")
    if vram_gb < 8:
        print("Warning: Unlimited-OCR usually needs about 8 GB VRAM. This GPU may run out of memory.")
    return "cuda"


def load_model(model_dir: str):
    import torch
    from transformers import AutoModel, AutoTokenizer

    print(f"Loading model: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_dir,
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=torch.bfloat16,
    )
    return tokenizer, model.eval().cuda()


def save_text(output_dir: str, name: str, text: str | None) -> None:
    if not text:
        return
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"{name}.md")
    with open(out_file, "w", encoding="utf-8") as handle:
        handle.write(text)
    print(f"  wrote {out_file}")


def infer_one(model, tokenizer, image_path: str, output_dir: str, image_mode: str, max_length: int):
    import torch

    mode = IMAGE_MODES[image_mode]
    name = os.path.splitext(os.path.basename(image_path))[0]
    t0 = time.time()
    with torch.no_grad():
        text = model.infer(
            tokenizer,
            prompt=PROMPT_SINGLE,
            image_file=image_path,
            output_path=output_dir,
            base_size=mode["base_size"],
            image_size=mode["image_size"],
            crop_mode=mode["crop_mode"],
            max_length=max_length,
            no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
            ngram_window=mode["ngram_window"],
            save_results=True,
        )
    print(f"  {name}: {time.time() - t0:.1f}s")
    if isinstance(text, str):
        save_text(output_dir, name, text)


def infer_multi(model, tokenizer, image_paths: list[str], output_dir: str, max_length: int):
    import torch

    mode = IMAGE_MODES["base"]
    t0 = time.time()
    with torch.no_grad():
        text = model.infer_multi(
            tokenizer,
            prompt=PROMPT_MULTI,
            image_files=image_paths,
            output_path=output_dir,
            image_size=mode["image_size"],
            max_length=max_length,
            no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
            ngram_window=mode["ngram_window"],
            save_results=True,
        )
    print(f"  multi-page ({len(image_paths)} pages): {time.time() - t0:.1f}s")
    if isinstance(text, str):
        save_text(output_dir, "multi_page", text)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local Unlimited-OCR inference on an NVIDIA GPU (Transformers, no SGLang).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_dir", default="baidu/Unlimited-OCR")
    parser.add_argument("--image_dir", default="", help="Directory of images")
    parser.add_argument("--pdf", default="", help="PDF file to parse")
    parser.add_argument("--output_dir", default="./outputs")
    parser.add_argument("--image_mode", choices=tuple(IMAGE_MODES), default="gundam")
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--max_pages", type=int, default=0, help="Limit PDF pages (0 = all)")
    parser.add_argument("--max_images", type=int, default=0, help="Limit images (0 = all)")
    parser.add_argument(
        "--per-page",
        action="store_true",
        help="Parse a PDF one page at a time instead of one-shot multi-page",
    )
    parser.add_argument("--check-gpu", action="store_true", help="Print GPU info and exit")
    parser.add_argument("--dpi", type=int, default=PDF_DPI)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    require_cuda()
    if args.check_gpu:
        return

    if args.pdf:
        images = pdf_to_images(args.pdf, dpi=args.dpi, max_pages=args.max_pages)
        mode = "pdf_pages"
    elif args.image_dir:
        images = collect_images(args.image_dir, max_images=args.max_images)
        mode = "dataset_images"
    else:
        raise SystemExit("Pass --pdf or --image_dir (or --check-gpu).")

    if not images:
        raise SystemExit("No input images found.")

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Mode: {mode}, images={len(images)}, image_mode={args.image_mode}")
    tokenizer, model = load_model(args.model_dir)

    if args.pdf and not args.per_page:
        infer_multi(model, tokenizer, images, args.output_dir, args.max_length)
        return

    for image_path in images:
        infer_one(model, tokenizer, image_path, args.output_dir, args.image_mode, args.max_length)


if __name__ == "__main__":
    main()
