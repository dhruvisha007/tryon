import base64
import unittest
from unittest.mock import patch, MagicMock

import handler

class TestHandler(unittest.TestCase):
    def test_missing_prompt(self):
        job = {"input": {"image": "base64data..."}}
        res = handler.handler(job)
        self.assertEqual(res["status"], "error")
        self.assertIn("prompt is required", res["message"])

    def test_missing_image(self):
        job = {"input": {"prompt": "a prompt"}}
        res = handler.handler(job)
        self.assertEqual(res["status"], "error")
        self.assertIn("image is required", res["message"])

    @patch("handler.load_pipeline")
    def test_invalid_image_decoding(self, mock_load):
        # Invalid base64 that causes PIL error
        job = {"input": {"prompt": "a prompt", "image": "invalid_base64_data_that_fails"}}
        res = handler.handler(job)
        self.assertEqual(res["status"], "error")
        self.assertIn("Image decoding failed", res["message"])

    @patch("handler.load_pipeline")
    @patch("handler.QwenImageEditPlusPipeline")
    def test_valid_request_structure(self, mock_pipe_class, mock_load):
        # Create a valid 1x1 image base64
        import io
        from PIL import Image
        img = Image.new('RGB', (1, 1), color='black')
        buf = io.BytesIO()
        img.save(buf, format='JPEG')
        b64 = base64.b64encode(buf.getvalue()).decode()

        mock_pipe = MagicMock()
        mock_load.return_value = mock_pipe
        
        # Mock result of pipeline
        mock_result = MagicMock()
        mock_result.images = [img]
        mock_pipe.return_value = mock_result

        job = {"input": {"prompt": "a prompt", "image": b64}}
        res = handler.handler(job)
        
        self.assertEqual(res["status"], "success")
        self.assertIn("image", res)
        self.assertTrue(isinstance(res["image"], str))
        self.assertIn("meta", res)

if __name__ == "__main__":
    unittest.main()
