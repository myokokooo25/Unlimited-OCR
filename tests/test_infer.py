"""Unit tests for infer.py helpers.

These cover the batch-inference planner and HTTP helpers. They do not
require a GPU, SGLang, or model weights.

Run with: python -m pytest tests/ -v
"""

from __future__ import annotations

import argparse
import base64
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from unittest.mock import MagicMock, patch

import pytest
import requests

import infer


PNG_1X1 = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
    b"\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00"
    b"\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _write_png(path) -> str:
    path.write_bytes(PNG_1X1)
    return str(path)


def _args(**overrides) -> argparse.Namespace:
    values = {
        "image_dir": "",
        "pdf": "",
        "output_dir": "./outputs",
        "concurrency": 8,
        "gpu": "0",
        "model_dir": "baidu/Unlimited-OCR",
        "image_mode": "gundam",
        "server_log": "./log/sglang_server.log",
        "dry_run": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class TestEncodeImage:
    def test_png_mime(self, tmp_path):
        result = infer.encode_image(_write_png(tmp_path / "test.png"))
        assert result["type"] == "image_url"
        assert result["image_url"]["url"].startswith("data:image/png;base64,")

    def test_jpg_mime(self, tmp_path):
        jpg_file = tmp_path / "test.jpg"
        jpg_file.write_bytes(b"\xff\xd8\xff\xe0fake_jpeg")
        result = infer.encode_image(str(jpg_file))
        assert "data:image/jpeg;base64," in result["image_url"]["url"]

    def test_jpeg_mime(self, tmp_path):
        jpeg_file = tmp_path / "test.jpeg"
        jpeg_file.write_bytes(b"\xff\xd8\xff\xe0fake_jpeg")
        result = infer.encode_image(str(jpeg_file))
        assert "data:image/jpeg;base64," in result["image_url"]["url"]

    def test_webp_mime(self, tmp_path):
        webp_file = tmp_path / "test.webp"
        webp_file.write_bytes(b"RIFF\x00\x00\x00\x00WEBP")
        result = infer.encode_image(str(webp_file))
        assert "data:image/webp;base64," in result["image_url"]["url"]

    def test_base64_content(self, tmp_path):
        content = b"hello world test content"
        img_file = tmp_path / "test.png"
        img_file.write_bytes(content)
        result = infer.encode_image(str(img_file))
        decoded = base64.b64decode(result["image_url"]["url"].split(",", 1)[1])
        assert decoded == content


class TestBuildContent:
    def test_structure(self, tmp_path):
        result = infer.build_content(_write_png(tmp_path / "img.png"))
        assert len(result) == 2
        assert result[0] == {"type": "text", "text": infer.PROMPT}
        assert result[1]["type"] == "image_url"


class TestServerReady:
    @patch("infer.requests.get")
    def test_healthy_server(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200)
        assert infer.server_ready("http://localhost:10000") is True
        mock_get.assert_called_once_with("http://localhost:10000/health", timeout=5)

    @patch("infer.requests.get")
    def test_unhealthy_server(self, mock_get):
        mock_get.return_value = MagicMock(status_code=500)
        assert infer.server_ready("http://localhost:10000") is False

    @patch("infer.requests.get")
    def test_connection_error(self, mock_get):
        mock_get.side_effect = requests.RequestException("Connection refused")
        assert infer.server_ready("http://localhost:10000") is False


class TestCollectDatasetImages:
    def test_finds_images_and_skips_other_files(self, tmp_path):
        (tmp_path / "a.png").write_bytes(b"png")
        (tmp_path / "b.jpg").write_bytes(b"jpg")
        (tmp_path / "c.txt").write_bytes(b"txt")
        result = infer.collect_dataset_images(str(tmp_path))
        assert {os.path.basename(p) for p in result} == {"a.png", "b.jpg"}

    def test_supported_extensions(self, tmp_path):
        for ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
            (tmp_path / f"img{ext}").write_bytes(b"x")
        result = infer.collect_dataset_images(str(tmp_path))
        assert len(result) == 5

    def test_empty_directory(self, tmp_path):
        (tmp_path / "readme.txt").write_bytes(b"text")
        assert infer.collect_dataset_images(str(tmp_path)) == []

    def test_sorted_by_size_descending(self, tmp_path):
        (tmp_path / "small.png").write_bytes(b"x")
        (tmp_path / "medium.png").write_bytes(b"xxx")
        (tmp_path / "large.png").write_bytes(b"xxxxx")
        names = [os.path.basename(p) for p in infer.collect_dataset_images(str(tmp_path))]
        assert names == ["large.png", "medium.png", "small.png"]

    def test_nested_directories(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (tmp_path / "root.png").write_bytes(b"x")
        (sub / "nested.jpg").write_bytes(b"y")
        assert len(infer.collect_dataset_images(str(tmp_path))) == 2


class TestBuildJobs:
    def test_image_dir_mode(self, tmp_path):
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        (img_dir / "a.png").write_bytes(b"x")
        nested = img_dir / "nested"
        nested.mkdir()
        (nested / "b.png").write_bytes(b"y")
        out_dir = tmp_path / "outputs"

        jobs = infer.build_jobs(
            _args(image_dir=str(img_dir), output_dir=str(out_dir))
        )
        assert len(jobs) == 2
        outputs = {output_file for _, output_file in jobs}
        assert any(path.endswith("a.md") for path in outputs)
        assert any("nested__b.md" in path for path in outputs)

    def test_no_args_raises(self):
        with pytest.raises(ValueError, match="Either --image_dir or --pdf"):
            infer.build_jobs(_args())

    def test_output_dir_none(self, tmp_path):
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        (img_dir / "a.png").write_bytes(b"x")
        jobs = infer.build_jobs(_args(image_dir=str(img_dir), output_dir=None))
        assert len(jobs) == 1
        assert jobs[0][1] is None

    def test_pdf_mode(self, tmp_path):
        pdf_path = tmp_path / "sample.pdf"
        _write_pdf(pdf_path, pages=2)
        out_dir = tmp_path / "outputs"
        jobs = infer.build_jobs(_args(pdf=str(pdf_path), output_dir=str(out_dir)))
        assert len(jobs) == 2
        assert all(os.path.exists(image_path) for image_path, _ in jobs)
        assert jobs[0][1] == str(out_dir / "sample_page_0001.md")
        assert jobs[1][1] == str(out_dir / "sample_page_0002.md")


def _write_pdf(path, pages: int = 1) -> str:
    import fitz

    doc = fitz.open()
    for i in range(pages):
        page = doc.new_page(width=200, height=200)
        page.insert_text((20, 80), f"Page {i + 1}")
    doc.save(str(path))
    doc.close()
    return str(path)


class TestPdfToImages:
    def test_converts_each_page(self, tmp_path):
        pdf_path = _write_pdf(tmp_path / "doc.pdf", pages=2)
        images = infer.pdf_to_images(pdf_path, dpi=72)
        assert len(images) == 2
        assert images[0].endswith("page_0001.png")
        assert images[1].endswith("page_0002.png")
        for image_path in images:
            assert os.path.getsize(image_path) > 0


class TestCollectStreamSilent:
    def test_parses_sse_lines(self, tmp_path):
        lines = [
            'data: {"choices":[{"delta":{"content":"Hello"}}]}',
            'data: {"choices":[{"delta":{"content":" World"}}]}',
            "data: [DONE]",
        ]
        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = [line.encode() for line in lines]
        out_file = str(tmp_path / "out.md")
        result = infer.collect_stream_silent(mock_resp, out_file)
        assert result["tokens"] == 2
        assert result["text"] == "Hello World"
        assert open(out_file, encoding="utf-8").read() == "Hello World"

    def test_empty_response(self):
        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = [b"data: [DONE]"]
        result = infer.collect_stream_silent(mock_resp, None)
        assert result["tokens"] == 0
        assert result["text"] == ""

    def test_malformed_json_skipped(self):
        lines = [
            "data: not-valid-json",
            'data: {"choices":[{"delta":{"content":"ok"}}]}',
            "data: [DONE]",
        ]
        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = [line.encode() for line in lines]
        result = infer.collect_stream_silent(mock_resp, None)
        assert result["text"] == "ok"

    def test_ignores_blank_and_non_data_lines(self):
        lines = [
            "",
            "event: ping",
            'data: {"choices":[{"delta":{"content":"x"}}]}',
            "data: [DONE]",
        ]
        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = [line.encode() for line in lines]
        result = infer.collect_stream_silent(mock_resp, None)
        assert result["text"] == "x"


class TestStopServer:
    def test_none_process(self):
        infer.stop_server(None)

    def test_terminates_process(self):
        mock_proc = MagicMock()
        infer.stop_server(mock_proc)
        mock_proc.terminate.assert_called_once()
        mock_proc.wait.assert_called_once_with(timeout=30)
        mock_proc._log_file.close.assert_called_once()

    def test_kills_if_terminate_times_out(self):
        mock_proc = MagicMock()
        mock_proc.wait.side_effect = [infer.subprocess.TimeoutExpired(cmd="sglang", timeout=30), None]
        infer.stop_server(mock_proc)
        mock_proc.kill.assert_called_once()


class TestStartServer:
    @patch("infer.server_ready", return_value=True)
    def test_reuses_existing_server(self, _mock_ready):
        assert infer.start_server(_args()) is None

    @patch("infer.server_ready", return_value=False)
    @patch("infer.subprocess.Popen")
    def test_raises_when_process_exits_early(self, mock_popen, _mock_ready, tmp_path):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 1
        mock_popen.return_value = mock_proc
        args = _args(server_log=str(tmp_path / "sglang.log"))
        with pytest.raises(RuntimeError, match="exited early"):
            infer.start_server(args)


class TestParseArgs:
    def test_defaults(self):
        with patch("sys.argv", ["infer.py"]):
            args = infer.parse_args()
        assert args.image_dir == ""
        assert args.pdf == ""
        assert args.output_dir == "./outputs"
        assert args.concurrency == 8
        assert args.image_mode == "gundam"
        assert args.dry_run is False

    def test_image_mode_choices(self):
        with patch("sys.argv", ["infer.py", "--image_mode", "base"]):
            assert infer.parse_args().image_mode == "base"

    def test_rejects_invalid_image_mode(self):
        with patch("sys.argv", ["infer.py", "--image_mode", "turbo"]):
            with pytest.raises(SystemExit):
                infer.parse_args()

    def test_dry_run_flag(self):
        with patch("sys.argv", ["infer.py", "--dry-run", "--image_dir", "./imgs"]):
            args = infer.parse_args()
        assert args.dry_run is True
        assert args.image_dir == "./imgs"


class TestInferOne:
    def test_success(self, tmp_path):
        image_path = _write_png(tmp_path / "page.png")
        output_file = str(tmp_path / "out.md")
        lines = [
            'data: {"choices":[{"delta":{"content":"parsed"}}]}',
            "data: [DONE]",
        ]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.iter_lines.return_value = [line.encode() for line in lines]

        with patch("infer.get_ngram_processor_str", return_value="processor"), patch(
            "infer.requests.post", return_value=mock_resp
        ) as mock_post:
            result = infer.infer_one(image_path, output_file, _args(), idx=1)

        assert result["text"] == "parsed"
        payload = json.loads(mock_post.call_args.kwargs["data"])
        assert payload["model"] == infer.SERVED_MODEL_NAME
        assert payload["images_config"]["image_mode"] == "gundam"
        assert payload["custom_params"] == {
            "ngram_size": infer.NO_REPEAT_NGRAM_SIZE,
            "window_size": infer.NGRAM_WINDOW,
        }
        assert open(output_file, encoding="utf-8").read() == "parsed"

    def test_retries_on_502_then_succeeds(self, tmp_path):
        image_path = _write_png(tmp_path / "page.png")
        bad = MagicMock(status_code=502)
        good = MagicMock(status_code=200)
        good.iter_lines.return_value = [
            b'data: {"choices":[{"delta":{"content":"ok"}}]}',
            b"data: [DONE]",
        ]
        with patch("infer.get_ngram_processor_str", return_value="processor"), patch(
            "infer.requests.post", side_effect=[bad, good]
        ), patch("infer.time.sleep"):
            result = infer.infer_one(image_path, None, _args(), idx=1)
        assert result["text"] == "ok"

    def test_returns_empty_after_retries_exhausted(self, tmp_path):
        image_path = _write_png(tmp_path / "page.png")
        with patch("infer.get_ngram_processor_str", return_value="processor"), patch(
            "infer.requests.post", side_effect=requests.RequestException("down")
        ), patch("infer.time.sleep"):
            result = infer.infer_one(image_path, None, _args(), idx=1)
        assert result == {"tokens": 0, "decode_time": 0, "text": ""}


class TestDryRun:
    def test_prints_jobs_without_starting_server(self, tmp_path, capsys):
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        _write_png(img_dir / "a.png")
        args = _args(image_dir=str(img_dir), output_dir=str(tmp_path / "out"), dry_run=True)
        with patch("infer.parse_args", return_value=args), patch("infer.start_server") as start:
            infer.main()
        start.assert_not_called()
        captured = capsys.readouterr().out
        assert "Dry run" in captured
        assert "a.png" in captured


class TestRunWithMockServer:
    def test_end_to_end_against_local_sse_server(self, tmp_path):
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        _write_png(img_dir / "doc.png")
        out_dir = tmp_path / "outputs"

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"ok")

            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                self.rfile.read(length)
                body = (
                    b'data: {"choices":[{"delta":{"content":"mock ocr"}}]}\n\n'
                    b"data: [DONE]\n\n"
                )
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format, *args):
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_address[1]}"
            args = _args(
                image_dir=str(img_dir),
                output_dir=str(out_dir),
                concurrency=1,
            )
            with patch("infer.SERVER_URL", url), patch(
                "infer.get_ngram_processor_str", return_value="processor"
            ):
                infer.run(args)
            outputs = list(out_dir.glob("*.md"))
            assert len(outputs) == 1
            assert outputs[0].read_text(encoding="utf-8") == "mock ocr"
        finally:
            server.shutdown()
            server.server_close()


class TestConstants:
    def test_server_constants(self):
        assert infer.PORT == 10000
        assert infer.HOST == "0.0.0.0"
        assert "10000" in infer.SERVER_URL

    def test_inference_constants(self):
        assert infer.TEMPERATURE == 0
        assert infer.NO_REPEAT_NGRAM_SIZE == 35
        assert infer.NGRAM_WINDOW == 128
        assert infer.CONTEXT_LENGTH == 32768
        assert infer.MAX_RETRIES == 5
        assert infer.REQUEST_TIMEOUT > 0
        assert infer.PDF_DPI == 300
        assert infer.SERVED_MODEL_NAME == "Unlimited-OCR"
