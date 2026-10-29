"""Model setting validation and secret-store behavior without persisting real keys."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.utils import model_settings


class ModelSettingsTests(unittest.TestCase):
    def test_rejects_provider_and_endpoint_changes(self) -> None:
        with self.assertRaisesRegex(ValueError, "must use"):
            model_settings._validate("deepseek", {"provider": "other"})
        with self.assertRaisesRegex(ValueError, "api.deepseek.com"):
            model_settings._validate("deepseek", {"base_url": "https://example.com"})

    def test_moark_key_is_shared_without_plaintext_persistence(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "secrets.json"
            with patch.object(model_settings, "SECRET_PATH", path), \
                 patch.object(model_settings, "_protect", return_value="encrypted"), \
                 patch.object(model_settings, "_unprotect", return_value="secret-value"), \
                 patch.dict("os.environ", {"MOARK_API_KEY": ""}):
                model_settings.set_secret("embedding", "secret-value")
                content = path.read_text(encoding="utf-8")
                self.assertNotIn("secret-value", content)
                self.assertEqual(model_settings.get_api_key("reranker"), "secret-value")
                model_settings.delete_secret("vision")
                self.assertEqual(model_settings._read_json(path), {})


if __name__ == "__main__":
    unittest.main()
