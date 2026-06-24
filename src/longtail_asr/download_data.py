from huggingface_hub import hf_hub_download, snapshot_download

hf_hub_download(
    repo_id="coml/sWuggy",
    repo_type="dataset",
    filename="inftrain/en/gold.csv",
    local_dir="data",
)

hf_hub_download(
    repo_id="coml/sWuggy",
    repo_type="dataset",
    filename="inftrain/fr/gold.csv",
    local_dir="data",
)

snapshot_download(
    repo_id="coml/sWuggy",
    repo_type="dataset",
    allow_patterns="inftrain/en/frequencies/*",
    local_dir="data",
)

snapshot_download(
    repo_id="coml/sWuggy",
    repo_type="dataset",
    allow_patterns="inftrain/fr/frequencies/*",
    local_dir="data",
)
