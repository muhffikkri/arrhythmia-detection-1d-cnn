# =====================================================================
# FILE: src/analysis/plot_distribution_ptbxl.py
# PURPOSE: VISUALISASI DISTRIBUSI KELAS PTB-XL
#          BERDASARKAN MAPPING LABEL DI config_labels.py
# =====================================================================

import os
import sys
import ast

# Add project root directory to python path
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from src.config import config as cfg
from src.config import config_labels as label_cfg

# =====================================================================
# OUTPUT DIRECTORY
# =====================================================================

OUTPUT_DIR = os.path.join(cfg.BASE_DIR, "output", "distribution_ptbxl")
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.1)

print("=" * 70)
print("DISTRIBUSI KELAS PTB-XL (BERDASARKAN config_labels.py)")
print("=" * 70)

# =====================================================================
# LOAD PTB-XL DATABASE (RAW SCP CODES)
# =====================================================================

print("\n--> Loading PTB-XL database...")

ptb_db = pd.read_csv(cfg.PTBXL_CSV, index_col=0)
ptb_db["scp_codes"] = ptb_db["scp_codes"].apply(ast.literal_eval)

print(f"    Total records (raw) : {len(ptb_db):,}")

# =====================================================================
# MAP SETIAP SCP CODE KE TARGET CLASS
# =====================================================================

mapping = label_cfg.PTBXL_TO_TARGET_MAPPING

all_mapped_records = []
mapped_target_counts = {cls: 0 for cls in label_cfg.TARGET_CLASSES + ["Others", "Unmapped"]}

for ecg_id, row in ptb_db.iterrows():
    scp_dict = row["scp_codes"]

    # Kumpulkan semua target yang cocok dari SCP codes
    matched = set()
    for scp_code in scp_dict.keys():
        if scp_code in mapping:
            matched.add(mapping[scp_code])

    # Priority resolution (sama seperti proccess_ptbxl.py)
    if "AF" in matched:
        target = "AF"
    elif "Takikardia" in matched:
        target = "Takikardia"
    elif "Bradikardia" in matched:
        target = "Bradikardia"
    elif "Others" in matched:
        target = "Others"
    elif "Normal" in matched:
        target = "Normal"
    else:
        target = "Unmapped"

    all_mapped_records.append({
        "ecg_id": ecg_id,
        "target_class": target,
        "scp_codes": list(scp_dict.keys()),
        "patient_id": row.get("patient_id", None),
        "strat_fold": row.get("strat_fold", None),
    })

    mapped_target_counts[target] += 1

df_mapped = pd.DataFrame(all_mapped_records)

print(f"    Total mapped records: {len(df_mapped):,}")
print(f"\n    Distribusi setelah mapping:")
for cls, cnt in mapped_target_counts.items():
    print(f"      {cls:15s} : {cnt:>6,} ({cnt/len(df_mapped)*100:.1f}%)")

# =====================================================================
# 1. OVERALL CLASS DISTRIBUTION (SEMUA TERMASUK OTHERS + UNMAPPED)
# =====================================================================

print("\n--> Plot 1: Overall class distribution (semua kelas)...")

# Urutan kelas yang konsisten dengan config
order_all = label_cfg.TARGET_CLASSES + ["Others", "Unmapped"]
order_all = [c for c in order_all if c in mapped_target_counts and mapped_target_counts[c] > 0]

palette_all = {
    "Normal": "#2ecc71",
    "AF": "#e74c3c",
    "Takikardia": "#e67e22",
    "Bradikardia": "#3498db",
    "Others": "#95a5a6",
    "Unmapped": "#bdc3c7",
}

fig, ax = plt.subplots(figsize=(14, 6), dpi=300)

counts_all = [mapped_target_counts[c] for c in order_all]
colors_all = [palette_all.get(c, "#7f8c8d") for c in order_all]

bars = ax.bar(order_all, counts_all, color=colors_all, edgecolor="white", linewidth=1.2)

for bar, cnt in zip(bars, counts_all):
    pct = cnt / sum(counts_all) * 100
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + max(counts_all) * 0.01,
        f"{cnt:,}\n({pct:.1f}%)",
        ha="center", va="bottom",
        fontsize=10, fontweight="bold", color="#2c3e50"
    )

ax.set_title(
    "PTB-XL: Distribusi Kelas Setelah Mapping (Semua Record)",
    fontsize=14, fontweight="bold", pad=15
)
ax.set_xlabel("Target Class", fontsize=12, fontweight="bold")
ax.set_ylabel("Jumlah Sampel", fontsize=12, fontweight="bold")
ax.set_ylim(0, max(counts_all) * 1.18)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "1_overall_class_distribution.png"),
    dpi=300, bbox_inches="tight"
)
plt.close()
print(f"    Saved: 1_overall_class_distribution.png")

# =====================================================================
# 2. DISTRIBUTION HANYA 4 KELAS TARGET (YANG DIPAKAI TRAINING)
# =====================================================================

print("\n--> Plot 2: 4 target class distribution (training classes)...")

df_target = df_mapped[df_mapped["target_class"].isin(label_cfg.TARGET_CLASSES)].copy()
target_counts = df_target["target_class"].value_counts().reindex(
    label_cfg.TARGET_CLASSES, fill_value=0
)

fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

colors_target = [palette_all[c] for c in label_cfg.TARGET_CLASSES]
bars = ax.bar(
    label_cfg.TARGET_CLASSES,
    [target_counts[c] for c in label_cfg.TARGET_CLASSES],
    color=colors_target,
    edgecolor="white",
    linewidth=1.2
)

total = target_counts.sum()
for bar, cls_name in zip(bars, label_cfg.TARGET_CLASSES):
    cnt = target_counts[cls_name]
    pct = cnt / total * 100
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + max(target_counts) * 0.01,
        f"{cnt:,}\n({pct:.1f}%)",
        ha="center", va="bottom",
        fontsize=11, fontweight="bold", color="#2c3e50"
    )

ax.set_title(
    "PTB-XL: Distribusi 4 Kelas Target (Training)",
    fontsize=14, fontweight="bold", pad=15
)
ax.set_xlabel("Target Class", fontsize=12, fontweight="bold")
ax.set_ylabel("Jumlah Sampel", fontsize=12, fontweight="bold")
ax.set_ylim(0, max(target_counts) * 1.20)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "2_target_class_distribution.png"),
    dpi=300, bbox_inches="tight"
)
plt.close()
print(f"    Saved: 2_target_class_distribution.png")

# =====================================================================
# 3. PIE CHART - PROPORSI 4 KELAS TARGET
# =====================================================================

print("\n--> Plot 3: Pie chart proporsi 4 target class...")

fig, ax = plt.subplots(figsize=(8, 8), dpi=300)

sizes = [target_counts[c] for c in label_cfg.TARGET_CLASSES]
explode = [0.03] * len(label_cfg.TARGET_CLASSES)

wedges, texts, autotexts = ax.pie(
    sizes,
    labels=label_cfg.TARGET_CLASSES,
    autopct=lambda pct: f"{pct:.1f}%\n({int(round(pct/100.*sum(sizes))):,})",
    colors=[palette_all[c] for c in label_cfg.TARGET_CLASSES],
    explode=explode,
    startangle=140,
    textprops={"fontsize": 11},
    pctdistance=0.72,
    labeldistance=1.12,
)

for autotext in autotexts:
    autotext.set_fontsize(9)
    autotext.set_fontweight("bold")

ax.set_title(
    "PTB-XL: Proporsi 4 Kelas Target",
    fontsize=14, fontweight="bold", pad=20
)

plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "3_target_class_pie_chart.png"),
    dpi=300, bbox_inches="tight"
)
plt.close()
print(f"    Saved: 3_target_class_pie_chart.png")

# =====================================================================
# 4. DISTRIBUSI PER SCP CODE (RAW DIAGNOSTIC CODES)
# =====================================================================

print("\n--> Plot 4: Distribusi SCP codes asli (sebelum mapping)...")

scp_counter = {}
for scp_dict in ptb_db["scp_codes"]:
    for code in scp_dict.keys():
        scp_counter[code] = scp_counter.get(code, 0) + 1

scp_sorted = sorted(scp_counter.items(), key=lambda x: x[1], reverse=True)

# Group SCP codes berdasarkan target class mereka
scp_with_target = {}
for code, _ in scp_sorted:
    if code in mapping:
        scp_with_target[code] = mapping[code]
    else:
        scp_with_target[code] = "Unmapped"

scp_df = pd.DataFrame([
    {"SCP Code": code, "Count": cnt, "Target Class": scp_with_target[code]}
    for code, cnt in scp_sorted
])

fig, ax = plt.subplots(figsize=(14, 8), dpi=300)

# Ambil top 15 SCP codes
top_n = 15
scp_top = scp_df.head(top_n).copy()
scp_top = scp_top.sort_values("Count", ascending=True)

bar_colors = [palette_all.get(t, "#7f8c8d") for t in scp_top["Target Class"]]

bars = ax.barh(scp_top["SCP Code"], scp_top["Count"], color=bar_colors, edgecolor="white")

for bar, cnt, tgt in zip(bars, scp_top["Count"], scp_top["Target Class"]):
    ax.text(
        bar.get_width() + max(scp_top["Count"]) * 0.01,
        bar.get_y() + bar.get_height() / 2,
        f"{cnt:,}  [{tgt}]",
        ha="left", va="center",
        fontsize=9, color="#2c3e50"
    )

ax.set_title(
    f"PTB-XL: Top {top_n} SCP Codes (Raw Diagnostic)\nWarna = Target Class Setelah Mapping",
    fontsize=14, fontweight="bold", pad=15
)
ax.set_xlabel("Jumlah Record", fontsize=12, fontweight="bold")
ax.set_ylabel("SCP Code", fontsize=12, fontweight="bold")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "4_scp_code_distribution.png"),
    dpi=300, bbox_inches="tight"
)
plt.close()
print(f"    Saved: 4_scp_code_distribution.png")

# =====================================================================
# 5. DISTRIBUSI PER FOLD (TRAIN / VAL / TEST)
# =====================================================================

print("\n--> Plot 5: Distribusi per fold split...")

df_target_fold = df_target.copy()

def assign_split(fold):
    fold = int(fold)
    if fold in range(1, 9):
        return "Train (1-8)"
    elif fold == 9:
        return "Val (9)"
    elif fold == 10:
        return "Test (10)"
    return "Unknown"

df_target_fold["Split"] = df_target_fold["strat_fold"].apply(assign_split)

split_order = ["Train (1-8)", "Val (9)", "Test (10)"]
split_palette = {"Train (1-8)": "#1f77b4", "Val (9)": "#ff7f0e", "Test (10)": "#2ca02c"}

fig, ax = plt.subplots(figsize=(14, 7), dpi=300)

sns.countplot(
    data=df_target_fold,
    x="target_class",
    hue="Split",
    order=label_cfg.TARGET_CLASSES,
    hue_order=split_order,
    palette=split_palette,
    edgecolor="white",
    linewidth=1,
    ax=ax
)

for p in ax.patches:
    height = int(p.get_height()) if not np.isnan(p.get_height()) else 0
    if height > 0:
        ax.annotate(
            f"{height:,}",
            (p.get_x() + p.get_width() / 2., height),
            ha="center", va="bottom",
            fontsize=9, fontweight="bold", color="#2c3e50",
            xytext=(0, 3), textcoords="offset points"
        )

ax.set_title(
    "PTB-XL: Distribusi Kelas Target per Fold Split",
    fontsize=14, fontweight="bold", pad=15
)
ax.set_xlabel("Target Class", fontsize=12, fontweight="bold")
ax.set_ylabel("Jumlah Sampel", fontsize=12, fontweight="bold")
ax.legend(title="Split", title_fontsize=11, loc="upper right")

max_h = max([p.get_height() for p in ax.patches if not np.isnan(p.get_height())] + [100])
ax.set_ylim(0, max_h * 1.15)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "5_class_distribution_per_fold.png"),
    dpi=300, bbox_inches="tight"
)
plt.close()
print(f"    Saved: 5_class_distribution_per_fold.png")

# =====================================================================
# 6. HEATMAP: SCP CODE vs TARGET CLASS
# =====================================================================

print("\n--> Plot 6: Heatmap SCP code to target class mapping...")

scp_target_matrix = []
for code, cnt in scp_sorted[:20]:
    for tgt in label_cfg.TARGET_CLASSES + ["Others"]:
        val = cnt if scp_with_target.get(code) == tgt else 0
        scp_target_matrix.append({
            "SCP Code": code,
            "Target Class": tgt,
            "Count": val
        })

hm_df = pd.DataFrame(scp_target_matrix)
hm_pivot = hm_df.pivot_table(
    index="SCP Code", columns="Target Class", values="Count", aggfunc="sum", fill_value=0
)

# Filter hanya baris yang ada count > 0
hm_pivot = hm_pivot.loc[hm_pivot.sum(axis=1) > 0]

fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

sns.heatmap(
    hm_pivot,
    annot=True,
    fmt="d",
    cmap="YlOrRd",
    linewidths=0.5,
    linecolor="white",
    cbar_kws={"label": "Jumlah Record"},
    ax=ax
)

ax.set_title(
    "PTB-XL: SCP Code -> Target Class Mapping Matrix",
    fontsize=14, fontweight="bold", pad=15
)
ax.set_xlabel("Target Class (config_labels)", fontsize=12, fontweight="bold")
ax.set_ylabel("SCP Code (Raw)", fontsize=12, fontweight="bold")

plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "6_scp_to_target_heatmap.png"),
    dpi=300, bbox_inches="tight"
)
plt.close()
print(f"    Saved: 6_scp_to_target_heatmap.png")

# =====================================================================
# 7. TABEL RINGKASAN (CSV)
# =====================================================================

print("\n--> Saving summary table...")

summary_rows = []

for cls in label_cfg.TARGET_CLASSES:
    cnt = mapped_target_counts.get(cls, 0)
    pct = cnt / len(df_mapped) * 100
    scp_codes_in_class = [
        code for code, target in mapping.items() if target == cls
    ]
    summary_rows.append({
        "Target Class": cls,
        "Total Records": cnt,
        "Percentage": f"{pct:.2f}%",
        "Mapped SCP Codes": ", ".join(sorted(scp_codes_in_class))
    })

summary_df = pd.DataFrame(summary_rows)

summary_path = os.path.join(OUTPUT_DIR, "distribution_summary.csv")
summary_df.to_csv(summary_path, index=False)

print(f"    Saved: distribution_summary.csv")

# =====================================================================
# PRINT SUMMARY
# =====================================================================

print("\n" + "=" * 70)
print("RINGKASAN DISTRIBUSI PTB-XL (CONFIG LABELS MAPPING)")
print("=" * 70)

print(f"\nTotal Record (raw)      : {len(ptb_db):>8,}")
print(f"Total Mapped (4 kelas)  : {len(df_target):>8,}")
print(f"Total Others            : {mapped_target_counts.get('Others', 0):>8,}")
print(f"Total Unmapped          : {mapped_target_counts.get('Unmapped', 0):>8,}")

print("\nDistribusi 4 Kelas Target:")
print("-" * 45)
for cls in label_cfg.TARGET_CLASSES:
    cnt = target_counts[cls]
    pct = cnt / total * 100
    print(f"  {cls:15s} : {cnt:>6,} ({pct:5.1f}%)")
print("-" * 45)
print(f"  {'Total':15s} : {total:>6,} (100.0%)")

print(f"\nMapping reference (config_labels.py):")
print(f"  PTBXL_TO_TARGET_MAPPING = {len(mapping)} entries")
print(f"  Priority: AF > Takikardia > Bradikardia > Others > Normal")

print("\n" + "=" * 70)
print("OUTPUT FILES:")
print("=" * 70)
print(f"  {OUTPUT_DIR}")
for f in sorted(os.listdir(OUTPUT_DIR)):
    print(f"    -> {f}")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
