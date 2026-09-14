# FILE: dataset_loader.py
# PURPOSE:
# - Single, data-driven entry point for loading PTB-XL and Chapman datasets.
# - No hard-coded class mapping inside training/evaluation scripts.
# - LABEL_SCHEME == "mapped" -> shared 4-class scheme (config.TARGET_CLASSES).
# - LABEL_SCHEME == "native" -> original dataset label NAMES, resolved straight
#   from the dataset files (PTB-XL SCP codes, Chapman SNOMED-CT codes).

import os
import ast
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from src.config import config as cfg
from src.config import config_labels as label_cfg
from src.config.experiment_configs import Config

warnings.filterwarnings("ignore")

# =====================================================================
# MANIFEST PATHS
# =====================================================================
PTBXL_MANIFEST = os.path.join(cfg.RESAMPLE_BASE, "manifest_ptbxl.csv")
CHAPMAN_MANIFEST = os.path.join(cfg.RESAMPLE_BASE, "manifest_chapman.csv")

# =====================================================================
# NATIVE LABEL RESOLUTION
# =====================================================================

def _load_ptbxl_database():
    """Raw PTB-XL database with SCP diagnostic codes per ECG record."""
    if not os.path.exists(cfg.PTBXL_CSV):
        raise FileNotFoundError(
            f"PTB-XL database not found at {cfg.PTBXL_CSV}."
        )
    return pd.read_csv(cfg.PTBXL_CSV)


def _primary_scp_code(scp_raw):
    """Primary (highest-confidence) SCP code from the scp_codes dict string."""
    if pd.isna(scp_raw):
        return None
    try:
        codes = ast.literal_eval(scp_raw)
    except (ValueError, SyntaxError):
        return None
    if not isinstance(codes, dict) or not codes:
        return None
    return max(codes, key=codes.get)


def _primary_dx_code(dx_raw):
    """Primary (first-listed) SNOMED-CT code from a Chapman #Dx line."""
    if pd.isna(dx_raw):
        return None
    parts = [c.strip() for c in str(dx_raw).split(",") if c.strip()]
    return parts[0] if parts else None


def resolve_label(df, dataset, label_scheme):
    """
    Build the resolved `label` column and the `class_names` list.

    mapped -> cfg.CLASS_NAMES (dropped "Others" rows).
    native -> dataset-native primary label (SCP code / SNOMED code).
    """
    if label_scheme == "mapped":
        df = df[df["target_class"].isin(cfg.CLASS_NAMES)].reset_index(drop=True)
        df["label"] = df["target_class"]
        class_names = list(cfg.CLASS_NAMES)
        return df, class_names

    if label_scheme != "native":
        raise ValueError(
            f"Unknown LABEL_SCHEME '{label_scheme}'. Use 'mapped' or 'native'."
        )

    if dataset == "PTBXL":
        df = df.merge(
            _load_ptbxl_database()[["ecg_id", "scp_codes"]],
            on="ecg_id",
            how="left"
        )
        df["label"] = df["scp_codes"].map(_primary_scp_code)
        df = df[df["label"].notna()].reset_index(drop=True)
    elif dataset == "CHAPMAN":
        df["label"] = df["diagnostic_string"].map(_primary_dx_code)
        df = df[df["label"].notna()].reset_index(drop=True)
    else:
        raise ValueError(f"Unknown dataset '{dataset}'. Use 'PTBXL' or 'CHAPMAN'.")

    class_names = sorted(df["label"].astype(str).unique().tolist())
    return df, class_names


# =====================================================================
# MANIFEST LOADING
# =====================================================================

def load_manifest(dataset, label_scheme):
    """Load manifest CSV and resolve its label column per scheme."""
    manifest_path = PTBXL_MANIFEST if dataset == "PTBXL" else CHAPMAN_MANIFEST
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"{dataset} manifest not found at {manifest_path}. "
            "Run the corresponding preprocessing script first."
        )
    df = pd.read_csv(manifest_path)
    df["filename_npy"] = df["filename_npy"].astype(str)
    return resolve_label(df, dataset, label_scheme)[0]


# =====================================================================
# SPLIT ASSIGNMENT
# =====================================================================

def assign_splits(df, dataset, label_col="label", min_count=8):
    """
    Assign a 'split' column (train/val/test) to every row.
    PTB-XL uses its stratified folds; Chapman uses a stratified random split.
    Classes present in fewer than `min_count` rows are dropped with a warning
    (they cannot be stratified into train/val/test meaningfully).
    """
    if dataset == "PTBXL":
        fold = df["strat_fold"].astype(int)
        split = np.where(fold <= 8, "train", np.where(fold == 9, "val", "test"))
        return df.assign(split=split)

    counts = df[label_col].value_counts()
    rare = counts[counts < min_count]
    if len(rare) > 0:
        print(
            f"[DatasetLoader] Dropping {len(rare)} rare native label(s) "
            f"(count < {min_count}) that cannot be stratified: "
            f"{dict(rare)}"
        )
        df = df[~df[label_col].isin(rare.index)].reset_index(drop=True)

    train, rest = train_test_split(
        df,
        test_size=0.30,
        stratify=df[label_col],
        random_state=42
    )
    val, test = train_test_split(
        rest,
        test_size=0.50,
        stratify=rest[label_col],
        random_state=42
    )
    return pd.concat(
        [train.assign(split="train"), val.assign(split="val"), test.assign(split="test")]
    ).reset_index(drop=True)


# =====================================================================
# DATASET LOADER
# =====================================================================

class DatasetLoader:
    """Load PTB-XL or Chapman into X/y arrays under the requested label scheme."""

    def __init__(
        self,
        dataset="PTBXL",
        label_scheme="mapped",
        folder_key=None,
        max_samples=None
    ):
        self.dataset = dataset
        self.label_scheme = label_scheme
        self.folder_key = folder_key or self._default_folder_key()
        self.fs = cfg.folder_fs(self.folder_key)
        self.signal_len = cfg.TARGET_LEN.get(self.fs, 2500)
        self.max_samples = max_samples
        self.manifest = load_manifest(dataset, label_scheme)
        self.df = assign_splits(self.manifest, dataset)
        if label_scheme == "mapped":
            self.class_names = list(cfg.CLASS_NAMES)
        else:
            self.class_names = sorted(self.df["label"].astype(str).unique().tolist())

    def _default_folder_key(self):
        if self.dataset == "CHAPMAN":
            return Config.FOLDER_CHAPMAN
        return Config.FOLDER_PTBXL

    def _signal_dir(self):
        return cfg.SUB_FOLDERS[self.folder_key]

    def _signal_path(self, row):
        folder = self._signal_dir()
        return os.path.join(folder, row["filename_npy"])

    def _to_one_hot(self, labels):
        lookup = {name: i for i, name in enumerate(self.class_names)}
        y = np.zeros((len(labels), len(self.class_names)), dtype=np.float32)
        for i, name in enumerate(labels):
            idx = lookup.get(name)
            if idx is not None:
                y[i, idx] = 1.0
        return y

    def _load_split(self, split):
        sub = self.df[self.df["split"] == split]
        if self.max_samples is not None:
            sub = sub.sample(
                n=min(len(sub), self.max_samples),
                random_state=42
            )
        X, labels = [], []
        for _, row in sub.iterrows():
            path = self._signal_path(row)
            if not os.path.exists(path):
                continue
            try:
                X.append(np.load(path).astype(np.float32))
                labels.append(str(row["label"]))
            except Exception:
                continue
        if not X:
            raise FileNotFoundError(
                f"No usable '{split}' signals found under {self._signal_dir()}. "
                "Check folder_key and that preprocessing ran successfully."
            )
        return np.array(X, dtype=np.float32), self._to_one_hot(labels)

    def build_splits(self):
        """Return {X_train, y_train, X_val, y_val, X_test, y_test} arrays."""
        train = self._load_split("train")
        val = self._load_split("val")
        test = self._load_split("test")
        return {
            "X_train": train[0],
            "y_train": train[1],
            "X_val": val[0],
            "y_val": val[1],
            "X_test": test[0],
            "y_test": test[1],
        }

    def build_eval_split(self, split="test"):
        """Load only one split (used by cross-dataset evaluation)."""
        X, y = self._load_split(split)
        return X, y

    def print_summary(self):
        print("\n" + "=" * 70)
        print(f"Dataset       : {self.dataset}")
        print(f"Label scheme  : {self.label_scheme}")
        print(f"Signal folder : {self.folder_key} -> {self._signal_dir()}")
        print(f"Sampling rate : {self.fs} Hz | signal length = {self.signal_len}")
        print(f"# classes     : {len(self.class_names)}")
        print(f"Class names   : {self.class_names}")
        print("=" * 70)
        for split in ["train", "val", "test"]:
            counts = self.df[self.df["split"] == split]["label"].value_counts()
            print(f"\n[{split.upper()} split] total={int(counts.sum())}")
            for name, cnt in counts.items():
                print(f"  {name:<20s} {cnt:>6d}")


# =====================================================================
# CLI ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    import sys

    _dataset = sys.argv[1] if len(sys.argv) > 1 else "PTBXL"
    _scheme = sys.argv[2] if len(sys.argv) > 2 else "mapped"
    loader = DatasetLoader(dataset=_dataset, label_scheme=_scheme)
    loader.print_summary()