# =====================================================================
# FILE: run_statistical_tests.py
# FINAL VERSION — STATISTICAL SIGNIFICANCE ANALYSIS
# =====================================================================

import os
import warnings

import numpy as np
import pandas as pd

from statistical_tests import (
    run_wilcoxon_test,
    generate_confidence_interval_report
)

import config as cfg

warnings.filterwarnings("ignore")

# =====================================================================
# OUTPUT DIRECTORY
# =====================================================================

STATS_DIR = os.path.join(
    cfg.OUTPUT_DIR,
    "statistical_tests"
)

os.makedirs(STATS_DIR, exist_ok=True)

# =====================================================================
# INPUT CSV
# =====================================================================

MASTER_CSV = os.path.join(
    cfg.OUTPUT_DIR,
    "experiments",
    "master_research_metrics.csv"
)

if not os.path.exists(MASTER_CSV):
    raise FileNotFoundError(
        f"master_research_metrics.csv tidak ditemukan:\n{MASTER_CSV}"
    )

# =====================================================================
# LOAD DATA
# =====================================================================

print("=" * 80)
print("STATISTICAL SIGNIFICANCE ANALYSIS")
print("=" * 80)

df = pd.read_csv(MASTER_CSV)

print(f"\nLoaded Rows : {len(df)}")

# =====================================================================
# FILTER ONLY PTB-XL INTERNAL RESULTS
# =====================================================================

df_internal = df[
    ~df["Experiment"].str.contains("Chapman", na=False)
].copy()

# =====================================================================
# DISPLAY AVAILABLE EXPERIMENTS
# =====================================================================

print("\nAvailable Experiments:")
for exp in sorted(df_internal["Experiment"].unique()):
    print(f" - {exp}")

# =====================================================================
# HELPER
# =====================================================================

def get_metric_array(df, exp_keyword, metric_name):

    rows = df[
        df["Experiment"].str.contains(
            exp_keyword,
            case=False,
            na=False
        )
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
print("CONFIDENCE INTERVAL REPORTS")
print("=" * 80)

ci_results = []

TARGET_EXPERIMENTS = [
    "PHASE1",
    "PHASE2",
    "PHASE3",
    "PHASE4"
]

for exp in TARGET_EXPERIMENTS:

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
            "Mean": mean_score,
            "Std": std_score,
            "CI95": ci,
            "Publication_Format":
                f"{mean_score:.3f} ± {ci:.3f}"
        })

    except Exception as e:
        print(f"\n[WARNING] {exp} skipped -> {e}")

# Save CI results
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
print("=" * 80)

wilcoxon_results = []

COMPARISONS = [

    # Representation Study
    (
        "PHASE1_100to250_clean",
        "PHASE1_500to250_clean"
    ),

    # Architecture Study
    (
        "PHASE2_Balanced",
        "PHASE2_Balanced_v2"
    ),

    (
        "PHASE2_Balanced",
        "PHASE2_Local_Focused"
    ),

    # Temporal Modeling
    (
        "PHASE3_Pure_CNN",
        "PHASE3_CNN_BiLSTM"
    ),

    (
        "PHASE3_Pure_CNN",
        "PHASE3_CNN_Attention"
    ),

    # Data Engineering
    (
        "PHASE4_Baseline",
        "PHASE4_SMOTE_TOMEK_Augmentation"
    )
]

for exp_a, exp_b in COMPARISONS:

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

        # Proteksi minimum sample
        min_len = min(
            len(scores_a),
            len(scores_b)
        )

        scores_a = scores_a[:min_len]
        scores_b = scores_b[:min_len]

        stat, p_value = run_wilcoxon_test(
            scores_a,
            scores_b,
            metric_name="Macro_F1"
        )

        wilcoxon_results.append({
            "Experiment_A": exp_a,
            "Experiment_B": exp_b,
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

# Save Wilcoxon
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

    summary_rows.append({
        "Experiment": exp,
        "Macro_F1_Mean":
            rows["Macro_F1"].mean(),

        "Macro_F1_STD":
            rows["Macro_F1"].std(),

        "Balanced_Accuracy_Mean":
            rows["Balanced_Accuracy"].mean(),

        "Macro_AUROC_Mean":
            rows["Macro_AUROC"].mean()
    })

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