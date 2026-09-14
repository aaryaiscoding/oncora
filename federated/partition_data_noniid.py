"""
partition_data_noniid.py
Replaces split_data.py's EVEN split with a controllable NON-IID split,
using the standard Dirichlet-distribution partitioning method from FL
literature (Hsu et al., "Measuring the Effects of Non-Identical Data
Distribution for Federated Visual Classification", 2019 — this is the
same method most FedAvg/FedProx comparison papers use to synthesize
realistic hospital-like heterogeneity from a single labeled dataset).

WHY THIS MATTERS FOR YOUR PAPER:
split_data.py's original approach (images[i::NUM_HOSPITALS]) gives every
hospital an almost identical class distribution — that's an IID split,
and IID data is the ONE setting where plain FedAvg already works fine.
It can't demonstrate FedProx's actual advantage. Real hospitals are non-
IID: one hospital sees more glioma referrals, another specializes in
pituitary cases, scanner/protocol differences skew the imaging
characteristics per class, etc. This script simulates that.

HOW IT WORKS:
For each class c, draw a proportion vector from Dir(alpha) across the
N hospitals, then assign that class's images to hospitals according to
those proportions.
  - alpha -> infinity: proportions all equal -> converges to the old
    even/IID split.
  - alpha -> 0: each class's images concentrate almost entirely in ONE
    hospital -> extreme non-IID (rarely realistic, but a useful stress
    test).
  - alpha ~ 0.3-0.7: moderate skew, the commonly used realistic setting
    in FL papers, and the one most defensible to cite in your writeup.

OUTPUTS:
  1. Physically partitioned folders, same structure as split_data.py
     (datasets/federated/hospital_i/<class>/...), so client.py and
     server.py need ZERO changes.
  2. A stacked bar chart (docs/partition_distribution.png) showing class
     counts per hospital — this IS your "prove the partition is actually
     uneven" figure for the paper/viva.
  3. A printed table of per-hospital class counts + total, for the
     appendix or a results table.

Run:
    python -m federated.partition_data_noniid --alpha 0.5
    python -m federated.partition_data_noniid --alpha 0.5 --num-hospitals 3
"""

import argparse
import os
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
SOURCE_DIR = _PROJECT_ROOT / "datasets" / "Training"
OUTPUT_DIR = _PROJECT_ROOT / "datasets" / "federated"
DOCS_DIR = _PROJECT_ROOT / "docs"
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]


def dirichlet_partition(class_images: dict, num_hospitals: int, alpha: float, seed: int) -> dict:
    """
    class_images: {class_name: [list of image filenames]}
    Returns: {hospital_idx: {class_name: [assigned filenames]}}
    """
    rng = np.random.default_rng(seed)
    assignment = {h: defaultdict(list) for h in range(num_hospitals)}

    for cls, images in class_images.items():
        images = list(images)
        rng.shuffle(images)
        n = len(images)

        # One proportion vector per class — this is what makes hospitals
        # differ in their CLASS MIX, not just their total volume.
        proportions = rng.dirichlet(alpha=[alpha] * num_hospitals)

        # Convert proportions to cumulative index cutpoints over the
        # shuffled image list for this class.
        cutpoints = (np.cumsum(proportions) * n).astype(int)
        cutpoints[-1] = n  # guard against rounding leaving stragglers unassigned

        start = 0
        for h, end in enumerate(cutpoints):
            assignment[h][cls] = images[start:end]
            start = end

    return assignment


def write_partition(assignment: dict, output_dir: str) -> None:
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    for h, class_dict in assignment.items():
        for cls, images in class_dict.items():
            dest_dir = os.path.join(output_dir, f"hospital_{h + 1}", cls)
            os.makedirs(dest_dir, exist_ok=True)
            src_dir = os.path.join(SOURCE_DIR, cls)
            for img in images:
                shutil.copy(os.path.join(src_dir, img), os.path.join(dest_dir, img))


def plot_distribution(assignment: dict, alpha: float, save_path: str) -> None:
    num_hospitals = len(assignment)
    hospital_labels = [f"Hospital {h + 1}" for h in range(num_hospitals)]

    counts = np.array([
        [len(assignment[h][cls]) for cls in CLASSES]
        for h in range(num_hospitals)
    ])  # shape: (num_hospitals, num_classes)

    fig, ax = plt.subplots(figsize=(8, 5))
    bottom = np.zeros(num_hospitals)
    colors = ["#b31b1b", "#111111", "#1e5631", "#8a6d1f"]

    for i, cls in enumerate(CLASSES):
        ax.bar(hospital_labels, counts[:, i], bottom=bottom, label=cls, color=colors[i])
        bottom += counts[:, i]

    ax.set_ylabel("Number of images")
    ax.set_title(f"Non-IID class distribution per hospital (Dirichlet alpha={alpha})")
    ax.legend(title="Class")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"Saved distribution chart to {save_path}")


def print_table(assignment: dict) -> None:
    print(f"\n{'Hospital':<12}" + "".join(f"{c:>14}" for c in CLASSES) + f"{'Total':>10}")
    for h in sorted(assignment):
        counts = [len(assignment[h][cls]) for cls in CLASSES]
        print(f"Hospital {h+1:<3}" + "".join(f"{c:>14}" for c in counts) + f"{sum(counts):>10}")


def main():
    parser = argparse.ArgumentParser(description="Non-IID Dirichlet partition of the MRI dataset.")
    parser.add_argument("--alpha", type=float, default=0.5,
                         help="Dirichlet concentration. Lower = more skewed/non-IID. "
                              "Try 100 for ~IID, 0.5 for moderate, 0.1 for extreme.")
    parser.add_argument("--num-hospitals", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    class_images = {}
    for cls in CLASSES:
        class_dir = os.path.join(SOURCE_DIR, cls)
        class_images[cls] = os.listdir(class_dir)

    assignment = dirichlet_partition(class_images, args.num_hospitals, args.alpha, args.seed)

    print_table(assignment)
    write_partition(assignment, OUTPUT_DIR)
    plot_distribution(
        assignment, args.alpha,
        os.path.join(DOCS_DIR, f"partition_distribution_alpha{args.alpha}.png"),
    )


if __name__ == "__main__":
    main()
