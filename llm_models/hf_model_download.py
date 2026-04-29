from huggingface_hub import snapshot_download


snapshot_download(
    repo_id="BAAI/bge-small-zh-v1.5",
    local_dir=r"E:\hf_models\models\bge-small-zh-v1.5"
)