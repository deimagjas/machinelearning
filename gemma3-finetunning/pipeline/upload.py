from huggingface_hub import upload_folder


def upload_model(folder_path: str, repo_id: str):
    return upload_folder(folder_path=folder_path, repo_id=repo_id)
