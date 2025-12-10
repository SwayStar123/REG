
import argparse
import os
from typing import Dict, Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets import Dataset

from dataset import CustomDataset


def customdataset_generator(data_dir: str, batch_size: int, num_workers: int) -> Iterator[Dict]:
    """Yield examples from CustomDataset in large batches.

    This reuses all existing dataset logic (paths, labels, etc.) and
    only keeps one batch in memory at a time.
    """

    ds = CustomDataset(data_dir)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    for images, vae_moments, labels in dl:
        # images: [B, C, H, W] uint8 (keep CHW layout)
        # vae_moments: [B, ...]
        # labels: [B]
        images_np = images.numpy()
        vae_np = vae_moments.numpy().astype("float32")
        labels_np = labels.numpy().astype("int64")

        b = images_np.shape[0]
        for i in range(b):
            img = images_np[i]  # C, H, W

            yield {
                "image": img,
                "vae_moments": vae_np[i],
                "label": int(labels_np[i]),
            }


def build_and_push(data_dir: str, repo_id: str, private: bool, max_shard_size: str, batch_size: int, num_workers: int):
    ds = Dataset.from_generator(
        lambda: customdataset_generator(data_dir, batch_size=batch_size, num_workers=num_workers),
    )

    ds.push_to_hub(repo_id, private=private, max_shard_size=max_shard_size)


def parse_args():
    parser = argparse.ArgumentParser(description="Build and push HF dataset for REPA Imagenet-512")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Root directory containing 'images' and 'vae-sd' subfolders")
    parser.add_argument("--repo-id", type=str, default="SwayStar123/repa-imagenet-512-invae")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--max-shard-size", type=str, default="10GB")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_and_push(
        data_dir=args.data_dir,
        repo_id=args.repo_id,
        private=args.private,
        max_shard_size=args.max_shard_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )