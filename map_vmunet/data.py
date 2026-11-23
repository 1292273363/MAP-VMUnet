# SPDX-License-Identifier: Apache-2.0
# Adapted from VM-UNet's ISIC dataset and transforms for fixed, public split lists.
"""Paired ISIC images and masks, loaded in the order used by each split."""

import csv
from pathlib import PurePosixPath
import random


SPLITS = ("train", "val", "test")
NORMALIZATION = {
    "isic2017": {"train": (159.922, 28.871), "eval": (148.429, 25.748)},
    "isic2018": {"train": (157.561, 26.706), "eval": (149.034, 32.022)},
}


def read_splits(split_dir, data_root, dataset):
    """Validate membership and map repository-relative paths under data_root."""
    manifest_path = split_dir / "split_manifest.csv"
    manifest = {}
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            sample_id = row["sample_id"]
            if not sample_id or sample_id in manifest or row["split"] not in SPLITS:
                raise ValueError(f"Invalid or repeated manifest entry: {sample_id!r}")
            files = []
            for column, folder in (("image_path", "images"), ("mask_path", "masks")):
                path = PurePosixPath(row[column])
                parts = path.parts
                suffix = ".jpg" if dataset == "isic2017" and folder == "images" else ".png"
                if (
                    len(parts) != 5
                    or parts[:2] != ("data", dataset)
                    or parts[2] not in ("train", "val")
                    or parts[3] != folder
                    or path.suffix != suffix
                    or "\\" in row[column]
                ):
                    raise ValueError(f"Invalid {column} in split manifest: {path}")
                files.append(data_root.joinpath(*parts[1:]))
            manifest[sample_id] = (row["split"], *files)

    result = {}
    seen = set()
    for split in SPLITS:
        split_path = split_dir / f"{split}.txt"
        ids = split_path.read_text(encoding="utf-8-sig").splitlines()
        if not ids:
            raise ValueError(f"Empty split list: {split_path}")
        entries = []
        for sample_id in ids:
            if sample_id in seen or sample_id not in manifest or manifest[sample_id][0] != split:
                raise ValueError(f"Invalid or repeated sample {sample_id!r} in {split_path}")
            seen.add(sample_id)
            entries.append((sample_id, *manifest[sample_id][1:]))
        result[split] = entries
    if seen != manifest.keys():
        raise ValueError(f"Split lists do not cover {manifest_path}")
    return result


def check_files(splits):
    for entries in splits.values():
        for sample_id, image_path, mask_path in entries:
            for path in (image_path, mask_path):
                if not path.is_file():
                    raise FileNotFoundError(f"{sample_id}: {path} is missing. See the data instructions.")


class ISICDataset:
    def __init__(self, entries, dataset, input_size, training=False):
        self.entries = entries
        self.input_size = input_size
        self.training = training
        self.mean, self.std = NORMALIZATION[dataset]["train" if training else "eval"]
        self.rotation_angle = random.uniform(0, 360) if training else None

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        import numpy as np
        from PIL import Image
        import torch
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms import functional as TF

        sample_id, image_path, mask_path = self.entries[index]
        with Image.open(image_path) as handle:
            image = np.array(handle.convert("RGB"))
        with Image.open(mask_path) as handle:
            mask = np.expand_dims(np.array(handle.convert("L")), 2) / 255

        image = (image - self.mean) / self.std
        minimum = image.min()
        span = image.max() - minimum
        if span == 0:
            raise ValueError(f"{sample_id} is constant and cannot be min-max normalized")
        image, mask = (
            torch.tensor(value).permute(2, 0, 1)
            for value in ((image - minimum) / span * 255.0, mask)
        )
        if self.training:
            if random.random() < 0.5:
                image, mask = TF.hflip(image), TF.hflip(mask)
            if random.random() < 0.5:
                image, mask = TF.vflip(image), TF.vflip(mask)
            if random.random() < 0.5:
                image, mask = TF.rotate(image, self.rotation_angle), TF.rotate(mask, self.rotation_angle)
        return tuple(
            TF.resize(value, self.input_size, interpolation=InterpolationMode.BILINEAR, antialias=False)
            for value in (image, mask)
        )
