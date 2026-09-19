# =====================================================================
# FILE: run_statistical_tests.py
# STATISTICAL SIGNIFICANCE ANALYSIS — reads the unified master tracker
# =====================================================================

import os
import sys
import warnings

# Add the project root directory to the python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import pandas as pd

from src.evaluation.statistical_tests import (
    run_wilcoxon_test,
    generate_confidence_interval_report
)

from src.config import config as cfg

warnings.filterwarnings("ignore")

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

# =====================================================================
# OUTPUT DIRECTORY
# =====================================================================

STATS_DIR = os.path.join(
    cfg.OUTPUT_DIR,
    "statistical_tests"
)

os.makedirs(STATS_DIR, exist_ok=True)

# =====================================================================
# INPUT CSV — unified runner tracker
# =====================================================================

MASTER_CSV = os.path.join(
    cfg.OUTPUT_DIR,
    "research_experiments",
    "master_experiment_tracker.csv"
)

if not os.path.exists(MASTER_CSV):
    raise FileNotFoundError(
        f"master_experiment_tracker.csv tidak ditemukan:\n{MASTER_CSV}"
    )

# =====================================================================
# LOAD DATA
# =====================================================================

print("=" * 80)
print("STATISTICAL SIGNIFICANCE ANALYSIS")
print("=" * 80)

df = pd.read_csv(MASTER_CSV)

df = df[
    df["Experiment"].notna()
    & df["Experiment"].astype(str).str.strip().ne("")
].copy()

print(f"\nLoaded Rows : {len(df)}")

# =====================================================================
# FILTER ONLY INTERNAL-DATASET RESULTS (Train == Test)
# =====================================================================

if {"Train_Dataset", "Test_Dataset"}.issubset(df.columns):
    def is_internal(row):
        tr = row["Train_Dataset"]
        te = row["Test_Dataset"]
        if pd.isna(tr) and pd.isna(te):
            return True
        return tr == te
    df_internal = df[
        df.apply(is_internal, axis=1)
    ].copy()
else:
    df_internal = df[
        ~df["Experiment"].str.contains("Chapman", na=False)
    ].copy()

# =====================================================================
# DISPLAY AVAILABLE EXPERIMENTS
# =====================================================================

print("\nAvailable Internal Experiments:")
for exp in sorted(df_internal["Experiment"].unique()):
    n = int(
        df_internal["Experiment"].eq(exp).sum()
    )
    print(f" - {exp}  (n={n})")

# =====================================================================
# HELPER
# =====================================================================

ARCH_SUFFIXES = (
    "__Pure_CNN",
    "__CNN_BiLSTM",
    "__CNN_Attention",
    "__Pure_LSTM"
)


def strip_architecture(exp_name):
    for suffix in ARCH_SUFFIXES:
        if exp_name.endswith(suffix):
            return exp_name[: -len(suffix)], suffix
    return exp_name, ""


def get_metric_array(df, exp_keyword, metric_name):

    rows = df[
        df["Experiment"] == exp_keyword
    ]

    if len(rows) == 0:
        raise ValueError(
            f"Tidak ditemukan eksperimen: {exp_keyword}"
        )

    values = rows[metric_name].dropna().values

    return values


# =====================================================================
# 1. CONFIDENCE INTERVAL REPORTS
# =====================================================================

print("\n" + "=" * 80)
print("CONFIDENCE INTERVAL REPORTS (per experiment, Macro_F1)")
print("=" * 80)

ci_results = []

for exp in sorted(df_internal["Experiment"].unique()):

    try:

        scores = get_metric_array(
            df_internal,
            exp,
            "Macro_F1"
        )

        mean_score, std_score, ci = (
            generate_confidence_interval_report(
                scores,
                metric_name=f"{exp} Macro_F1"
            )
        )

        ci_results.append({
            "Experiment": exp,
            "N": len(scores),
            "Mean": mean_score,
            "Std": std_score,
            "CI95": ci,
            "Publication_Format":
                f"{mean_score:.3f} ± {ci:.3f}"
        })

    except Exception as e:
        print(f"\n[WARNING] {exp} skipped -> {e}")

ci_df = pd.DataFrame(ci_results)

ci_csv = os.path.join(
    STATS_DIR,
    "confidence_interval_results.csv"
)

ci_df.to_csv(ci_csv, index=False)

print(f"\n✓ Saved: {ci_csv}")

# =====================================================================
# 2. WILCOXON COMPARISONS
# =====================================================================

print("\n" + "=" * 80)
print("WILCOXON SIGNED-RANK TESTS")
print("(pairs derived from runs sharing the same base experiment,")
print(" differing only in the architecture suffix)")
print("=" * 80)

wilcoxon_results = []

exp_counts = df_internal["Experiment"].value_counts()

base_groups = {}

for exp in df_internal["Experiment"].unique():

    base, arch = strip_architecture(exp)

    if not base:
        continue

    base_groups.setdefault(base, {})[exp] = int(exp_counts[exp])

for base, exps in base_groups.items():

    if len(exps) < 2:
        continue

    pairs = []

    exp_list = sorted(exps.keys())

    for i in range(len(exp_list)):
        for j in range(i + 1, len(exp_list)):
            pairs.append((exp_list[i], exp_list[j]))

    for exp_a, exp_b in pairs:

        if exps[exp_a] < 2 or exps[exp_b] < 2:
            continue

        try:

            scores_a = get_metric_array(
                df_internal,
                exp_a,
                "Macro_F1"
            )

            scores_b = get_metric_array(
                df_internal,
                exp_b,
                "Macro_F1"
            )

            min_len = min(
                len(scores_a),
                len(scores_b)
            )

            if min_len < 2:
                continue

            scores_a = scores_a[:min_len]
            scores_b = scores_b[:min_len]

            stat, p_value = run_wilcoxon_test(
                scores_a,
                scores_b,
                metric_name="Macro_F1"
            )

            wilcoxon_results.append({
                "Base": base,
                "Experiment_A": exp_a,
                "Experiment_B": exp_b,
                "N_Pairs": min_len,
                "Mean_A": np.mean(scores_a),
                "Mean_B": np.mean(scores_b),
                "Wilcoxon_Statistic": stat,
                "P_Value": p_value,
                "Significant": p_value < 0.05
            })

        except Exception as e:
            print(f"\n[WARNING] Comparison skipped:")
            print(f"{exp_a} vs {exp_b}")
            print(e)

wilcoxon_df = pd.DataFrame(
    wilcoxon_results
)

wilcoxon_csv = os.path.join(
    STATS_DIR,
    "wilcoxon_results.csv"
)

wilcoxon_df.to_csv(
    wilcoxon_csv,
    index=False
)

print(f"\n✓ Saved: {wilcoxon_csv}")

# =====================================================================
# 3. SUMMARY TABLE
# =====================================================================

summary_rows = []

for exp in sorted(df_internal["Experiment"].unique()):

    rows = df_internal[
        df_internal["Experiment"] == exp
    ]

    if len(rows) == 0:
        continue

    entry = {
        "Experiment": exp,
        "N": len(rows),
        "Macro_F1_Mean":
            rows["Macro_F1"].mean(),

        "Macro_F1_STD":
            rows["Macro_F1"].std(),

        "Balanced_Accuracy_Mean":
            rows["Balanced_Accuracy"].mean(),
    }

    if "Macro_AUROC" in rows.columns:
        entry["Macro_AUROC_Mean"] = rows["Macro_AUROC"].mean()
    else:
        entry["Macro_AUROC_Mean"] = float("nan")

    for col in ["Scheme", "Label_Scheme"]:
        if col in rows.columns:
            entry[col] = rows[col].iloc[0]

    summary_rows.append(entry)

summary_df = pd.DataFrame(summary_rows)

summary_csv = os.path.join(
    STATS_DIR,
    "experiment_summary.csv"
)

summary_df.to_csv(summary_csv, index=False)

print(f"\n✓ Saved: {summary_csv}")

# =====================================================================
# FINISHED
# =====================================================================

print("\n" + "=" * 80)
print("ALL STATISTICAL TESTS FINISHED")
print("=" * 80)