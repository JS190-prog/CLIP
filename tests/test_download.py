import hashlib
import importlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock


clip_module = importlib.import_module("clip.clip")


class DownloadTest(unittest.TestCase):
    def test_download_with_or_without_content_length(self):
        content = b"synthetic model payload"
        digest = hashlib.sha256(content).hexdigest()
        for headers in ({}, {"Content-Length": str(len(content))}):
            with self.subTest(headers=headers), tempfile.TemporaryDirectory() as root:
                response = io.BytesIO(content)
                response.info = lambda: headers
                url = f"https://example.invalid/{digest}/model.pt"
                with mock.patch.object(clip_module.urllib.request, "urlopen", return_value=response):
                    target = clip_module._download(url, root)
                self.assertEqual(Path(target).read_bytes(), content)

    def test_unknown_length_still_checks_checksum(self):
        with tempfile.TemporaryDirectory() as root:
            response = io.BytesIO(b"corrupt payload")
            response.info = lambda: {}
            url = f"https://example.invalid/{'0' * 64}/model.pt"
            with mock.patch.object(clip_module.urllib.request, "urlopen", return_value=response):
                with self.assertRaisesRegex(RuntimeError, "SHA256"):
                    clip_module._download(url, root)

    def test_verified_cached_download_does_not_request_network(self):
        content = b"cached model payload"
        digest = hashlib.sha256(content).hexdigest()
        with tempfile.TemporaryDirectory() as root:
            target = Path(root) / "model.pt"
            target.write_bytes(content)
            with mock.patch.object(clip_module.urllib.request, "urlopen") as request:
                self.assertEqual(clip_module._download(
                    f"https://example.invalid/{digest}/model.pt", root
                ), str(target))
            request.assert_not_called()


if __name__ == "__main__":
    unittest.main()
