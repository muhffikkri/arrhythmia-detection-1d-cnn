"""Display the Chapman class distribution using the configured label mapping."""

import argparse
import os
import sys
from collections import Counter

import pandas as pd


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import config as cfg
from src.config import config_labels as label_cfg


PRIORITY = ["AF", "Takikardia", "Bradikardia", "Others", "Normal"]


def map_chapman_classes(dx_string):
    """Map a Chapman diagnostic string to one target class."""
    matched_labels = {
        label_cfg.CHAPMAN_TO_TARGET_MAPPING[code.strip()]
        for code in dx_string.split(",")
        if code.strip() in label_cfg.CHAPMAN_TO_TARGET_MAPPING
    }

    for target_class in PRIORITY:
        if target_class in matched_labels:
            return target_class

    return None


def read_diagnostic_string(hea_path):
    """Read the Chapman #Dx field from a WFDB header."""
    with open(hea_path, encoding="utf-8") as header_file:
        for line in header_file:
            if line.startswith("#Dx:"):
                return line.split("#Dx:", 1)[1].strip()
    return ""


def collect_distribution(records_dir):
    """Return per-record mapping results and aggregate counts."""
    records = []

    for root, _, filenames in os.walk(records_dir):
        for filename in filenames:
            if not filename.endswith(".hea"):
                continue

            hea_path = os.path.join(root, filename)
            diagnostic_string = read_diagnostic_string(hea_path)
            target_class = map_chapman_classes(diagnostic_string) if diagnostic_string else None

            records.append(
                {
                    "record_id": os.path.splitext(filename)[0],
                    "diagnostic_string": diagnostic_string,
                    "target_class": target_class or "Unmapped",
                }
            )

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(
        description="Tampilkan distribusi kelas dataset Chapman berdasarkan config label."
    )
    parser.add_argument(
        "--records-dir",
        default=cfg.CHAPMAN_RECS,
        help="Direktori Chapman/WFDBRecords (default: dari config).",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(cfg.BASE_DIR, "output", "eda", "chapman_class_distribution.csv"),
        help="Lokasi CSV hasil (default: output/eda/chapman_class_distribution.csv).",
    )
    args = parser.parse_args()

    distribution_df = collect_distribution(args.records_dir)
    if distribution_df.empty:
        raise SystemExit(f"Tidak ditemukan file .hea di: {args.records_dir}")

    counts = Counter(distribution_df["target_class"])
    ordered_classes = [*label_cfg.TARGET_CLASSES, "Others", "Unmapped"]

    summary = pd.DataFrame(
        {
            "target_class": ordered_classes,
            "record_count": [counts.get(target_class, 0) for target_class in ordered_classes],
        }
    )
    summary = summary[summary["record_count"] > 0].reset_index(drop=True)
    summary["percentage"] = (summary["record_count"] / len(distribution_df) * 100).round(2)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    summary.to_csv(args.output, index=False)

    print("=" * 60)
    print("DISTRIBUSI KELAS DATASET CHAPMAN")
    print("=" * 60)
    print(f"Total file header : {len(distribution_df)}")
    print(f"Total mapped      : {len(distribution_df) - counts.get('Unmapped', 0)}")
    print(f"Total unmapped    : {counts.get('Unmapped', 0)}")
    print()
    print(summary.to_string(index=False))
    print(f"\nCSV tersimpan di: {args.output}")


if __name__ == "__main__":
    main()