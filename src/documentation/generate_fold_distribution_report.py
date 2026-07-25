# =====================================================================
# FILE: src/documentation/generate_fold_distribution_report.py
# PURPOSE: REPORT DOCUMENTATION TOOL — CLASS DISTRIBUTION PER FOLD SPLIT
# =====================================================================

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root directory to python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config import config as cfg

def generate_fold_distribution_report():
    print("=" * 80)
    print("GENERATING DATASET FOLD CLASS DISTRIBUTION REPORT")
    print("=" * 80)

    # Output Directory
    out_dir = os.path.join(cfg.OUTPUT_DIR, "documentation_reports")
    os.makedirs(out_dir, exist_ok=True)

    # Manifest Path
    manifest_path = os.path.join(cfg.RESAMPLE_BASE, "manifest_ptbxl.csv")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}. Please run preprocessing first.")

    df_manifest = pd.read_csv(manifest_path)
    df_filtered = df_manifest[df_manifest["target_class"].isin(cfg.CLASS_NAMES)].copy()

    # Assign Split Labels
    def assign_split(fold):
        fold = int(fold)
        if fold in range(1, 9):
            return "Train (Folds 1-8)"
        elif fold == 9:
            return "Validation (Fold 9)"
        elif fold == 10:
            return "Test (Fold 10)"
        return "Excluded"

    df_filtered["Split"] = df_filtered["strat_fold"].apply(assign_split)
    df_valid = df_filtered[df_filtered["Split"] != "Excluded"].copy()

    # Compute Summary Table
    pivot_table = pd.crosstab(
        df_valid["target_class"],
        df_valid["Split"],
        margins=True,
        margins_name="Total"
    ).reindex(index=cfg.CLASS_NAMES + ["Total"])

    report_rows = []
    total_samples = pivot_table.loc["Total", "Total"]

    for cls in cfg.CLASS_NAMES:
        tr_cnt = pivot_table.loc[cls, "Train (Folds 1-8)"]
        va_cnt = pivot_table.loc[cls, "Validation (Fold 9)"]
        te_cnt = pivot_table.loc[cls, "Test (Fold 10)"]
        tot_cnt = pivot_table.loc[cls, "Total"]

        tr_pct = (tr_cnt / pivot_table.loc["Total", "Train (Folds 1-8)"]) * 100
        va_pct = (va_cnt / pivot_table.loc["Total", "Validation (Fold 9)"]) * 100
        te_pct = (te_cnt / pivot_table.loc["Total", "Test (Fold 10)"]) * 100
        tot_pct = (tot_cnt / total_samples) * 100

        report_rows.append({
            "Class": cls,
            "Train Count": tr_cnt,
            "Train %": f"{tr_pct:.2f}%",
            "Val Count": va_cnt,
            "Val %": f"{va_pct:.2f}%",
            "Test Count": te_cnt,
            "Test %": f"{te_pct:.2f}%",
            "Total Count": tot_cnt,
            "Total %": f"{tot_pct:.2f}%"
        })

    # Add Total Row
    report_rows.append({
        "Class": "Total",
        "Train Count": pivot_table.loc["Total", "Train (Folds 1-8)"],
        "Train %": "100.00%",
        "Val Count": pivot_table.loc["Total", "Validation (Fold 9)"],
        "Val %": "100.00%",
        "Test Count": pivot_table.loc["Total", "Test (Fold 10)"],
        "Test %": "100.00%",
        "Total Count": total_samples,
        "Total %": "100.00%"
    })

    report_df = pd.DataFrame(report_rows)

    # Save CSV and Markdown Reports
    csv_path = os.path.join(out_dir, "class_fold_distribution.csv")
    md_path = os.path.join(out_dir, "class_fold_distribution.md")
    report_df.to_csv(csv_path, index=False)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# PTB-XL Dataset Fold Class Distribution Summary\n\n")
        f.write(report_df.to_markdown(index=False))
        f.write("\n")

    print("\n--- Summary Table ---")
    print(report_df.to_string(index=False))
    print(f"\n[SAVED] CSV report: {csv_path}")
    print(f"[SAVED] Markdown report: {md_path}")

    # =====================================================================
    # PUBLICATION-GRADE VISUALIZATION (300 DPI)
    # =====================================================================
    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)

    # Prepare data for Seaborn grouped bar plot
    plot_df = df_valid[df_valid["target_class"].isin(cfg.CLASS_NAMES)].copy()
    
    palette = {
        "Train (Folds 1-8)": "#1f77b4",
        "Validation (Fold 9)": "#ff7f0e",
        "Test (Fold 10)": "#2ca02c"
    }

    g = sns.countplot(
        data=plot_df,
        x="target_class",
        hue="Split",
        order=cfg.CLASS_NAMES,
        hue_order=["Train (Folds 1-8)", "Validation (Fold 9)", "Test (Fold 10)"],
        palette=palette,
        ax=ax
    )

    # Add exact count annotations on top of each bar
    for p in ax.patches:
        height = int(p.get_height()) if not np.isnan(p.get_height()) else 0
        if height > 0:
            ax.annotate(
                f"{height:,}",
                (p.get_x() + p.get_width() / 2., height),
                ha='center', va='bottom',
                fontsize=10, fontweight='bold', color='#2c3e50',
                xytext=(0, 3), textcoords='offset points'
            )

    ax.set_title("PTB-XL Stratified Fold Distribution Across Target Classes", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Target Arrhythmia Class", fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel("Sample Count (N)", fontsize=12, fontweight='bold', labelpad=10)
    ax.legend(title="Dataset Partition", title_fontsize='11', loc='upper right', frameon=True)
    
    # Adjust y-axis limit for annotations padding
    max_h = max([p.get_height() for p in ax.patches if not np.isnan(p.get_height())] + [100])
    ax.set_ylim(0, max_h * 1.12)

    plt.tight_layout()
    plot_path = os.path.join(out_dir, "class_fold_distribution.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"[SAVED] Distribution Plot: {plot_path}\n")

if __name__ == "__main__":
    generate_fold_distribution_report()
