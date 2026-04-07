from unittest.mock import MagicMock, patch

from pipeline.upload import upload_model


class TestUploadModel:
    @patch("pipeline.upload.upload_folder")
    def test_calls_upload_folder(self, mock_upload):
        # Given — a mocked upload_folder that returns a commit URL
        mock_upload.return_value = MagicMock(
            commit_url="https://huggingface.co/user/repo/commit/abc123"
        )

        # When — upload_model is called with a folder path and repo ID
        result = upload_model(
            folder_path="./fused_model",
            repo_id="user/my-model",
        )

        # Then — upload_folder receives the exact arguments and the commit URL is returned
        mock_upload.assert_called_once_with(
            folder_path="./fused_model",
            repo_id="user/my-model",
        )
        assert result.commit_url == "https://huggingface.co/user/repo/commit/abc123"
