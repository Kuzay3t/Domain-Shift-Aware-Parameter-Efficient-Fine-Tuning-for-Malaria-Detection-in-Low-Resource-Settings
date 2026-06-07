"""
download_datasets.py
====================
Downloads all datasets required to reproduce the DSA-LoRA experiments.

Datasets:
    1. NIH Malaria Detection Dataset (source domain — HIC)
       https://www.kaggle.com/datasets/shahriar26s/malaria-detection

    2. Lacuna Malaria Detection Challenge Dataset (target domain — LMIC)
       https://www.kaggle.com/datasets/rajsahu2004/lacuna-malaria-detection-dataset

    3. Stage 1 Trained Model Weights (source domain ViT-B/16)
       https://www.kaggle.com/datasets/gwaceee/miccai-dataset

    4. Stage 2 MMD Results (layer-wise domain shift scores)
       https://www.kaggle.com/datasets/gwaceee/miccaidata2

Usage:
    python download_datasets.py

Requirements:
    pip install kagglehub

    You must have a Kaggle account and API credentials configured.
    See: https://www.kaggle.com/docs/api#authentication
"""

import os
import kagglehub


DATASETS = [
    {
        "name": "NIH Malaria Detection Dataset (source domain)",
        "handle": "shahriar26s/malaria-detection",
        "role": "Source domain — clean HIC malaria cell images (Stage 1 training)",
        "env_var": "NIH_MALARIA_PATH",
    },
    {
        "name": "Lacuna Malaria Detection Challenge Dataset (target domain)",
        "handle": "rajsahu2004/lacuna-malaria-detection-dataset",
        "role": "Target domain — LMIC smartphone microscopy images (Stages 3 & 4)",
        "env_var": "LACUNA_MALARIA_PATH",
    },
    {
        "name": "Stage 1 Model Weights",
        "handle": "gwaceee/miccai-dataset",
        "role": "Pretrained source domain ViT-B/16 model (Stages 2, 3 & 4)",
        "env_var": "STAGE1_MODEL_PATH",
    },
    {
        "name": "Stage 2 MMD Results",
        "handle": "gwaceee/miccaidata2",
        "role": "Layer-wise MMD scores and LoRA layer selection (Stages 3 & 4)",
        "env_var": "STAGE2_MMD_PATH",
    },
]


def download_all():
    print("=" * 60)
    print("DSA-LoRA: Downloading all required datasets")
    print("=" * 60)

    paths = {}

    for i, dataset in enumerate(DATASETS, 1):
        print(f"\n[{i}/{len(DATASETS)}] {dataset['name']}")
        print(f"  Handle : {dataset['handle']}")
        print(f"  Role   : {dataset['role']}")

        try:
            path = kagglehub.dataset_download(dataset["handle"])
            paths[dataset["env_var"]] = path
            print(f"  Path   : {path}")
            print(f"  Status : ✓ Downloaded successfully")

        except Exception as e:
            print(f"  Status : ✗ Failed — {e}")
            print(f"  Fix    : Make sure your Kaggle API credentials are set up.")
            print(f"           Run: kaggle datasets download {dataset['handle']}")

    print("\n" + "=" * 60)
    print("Download complete. Dataset paths:")
    print("=" * 60)
    for env_var, path in paths.items():
        print(f"  {env_var}={path}")

    print("\nTo use these paths in your scripts, either:")
    print("  1. Pass them directly to kagglehub.dataset_download() as above")
    print("  2. Or set environment variables and read with os.environ.get()")

    return paths


if __name__ == "__main__":
    download_all()
