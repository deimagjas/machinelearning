from unittest.mock import MagicMock, patch

from pipeline.upload import upload_model


class TestUploadModel:
    @patch("pipeline.upload.upload_folder")
    def test_calls_upload_folder(self, mock_upload):
        mock_upload.return_value = MagicMock(
            commit_url="https://huggingface.co/user/repo/commit/abc123"
        )

        result = upload_model(
            folder_path="./fused_model",
            repo_id="user/my-model",
        )

        mock_upload.assert_called_once_with(
            folder_path="./fused_model",
            repo_id="user/my-model",
        )
        assert result.commit_url == "https://huggingface.co/user/repo/commit/abc123"
