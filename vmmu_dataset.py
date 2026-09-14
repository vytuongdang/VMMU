"""Load a VMMU split from Hugging Face (anvo25/vmmu) and cache its images locally.

The API callers send each image by file path, so the first time a split is used its images are
written to their `image_path` under the repo root (e.g. dataset/full/Math/...png); later runs reuse
them. The returned items have the dataset's fields, with `image` / `image_path` pointing at that file.

Also usable on its own to pre-download a split:
    python vmmu_dataset.py --split full_vqa
"""
import argparse
from pathlib import Path

HF_DATASET = "anvo25/vmmu"
SPLITS = [
    "full_vqa",
    "random_subset_vqa",
    "random_subset_ocr",
    "cropped_random_subset_vqa",
    "cropped_random_subset_vqa_description",
]
REPO_ROOT = Path(__file__).resolve().parent


def load_split(split, limit=None):
    from datasets import Image, load_dataset

    # decode=False: keep the original image bytes instead of decoding + re-encoding them
    ds = load_dataset(HF_DATASET, split=split).cast_column("image", Image(decode=False))
    rows = ds.select_columns([c for c in ds.column_names if c != "image"])
    n = len(ds) if limit is None else min(limit, len(ds))

    items, downloaded = [], 0
    for i in range(n):
        row = rows[i]
        local_path = REPO_ROOT / row["image_path"]  # e.g. dataset/full/Math/...png
        if not local_path.exists():
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(ds[i]["image"]["bytes"])
            downloaded += 1
        image_path = row["image_path"]
        items.append({"ID": row["ID"], "image": image_path, "image_path": image_path,
                      **{k: v for k, v in row.items() if k not in ("ID", "image_path")}})

    print(f"Loaded {len(items)} items from {HF_DATASET} (split={split}); "
          f"{downloaded} new images saved to dataset/")
    return items


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a VMMU split from Hugging Face and cache its images.")
    parser.add_argument("--split", choices=SPLITS, nargs="+", default=SPLITS)
    for split in parser.parse_args().split:
        load_split(split)
