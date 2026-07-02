# =====================================================================

# FILE: visualization.py

# =====================================================================

import os
import numpy as np
import matplotlib.pyplot as plt

# =====================================================================

# SAVE MISCLASSIFIED ECG

# =====================================================================

def save_misclassified_samples(
    X,
    y_true,
    y_pred,
    class_names,
    save_dir,
    max_samples=20
):

    mis_idx = np.where(
        y_true != y_pred
    )[0]

    save_dir = os.path.join(
        save_dir,
        "misclassified"
    )

    os.makedirs(
        save_dir,
        exist_ok=True
    )

    for i, idx in enumerate(mis_idx[:max_samples]):

        signal = X[idx]

        true_cls = class_names[
            y_true[idx]
        ]

        pred_cls = class_names[
            y_pred[idx]
        ]

        plt.figure(figsize=(14, 4))

        for ch in range(signal.shape[-1]):

            plt.plot(
                signal[:, ch],
                label=f"Lead {ch+1}"
            )

        plt.title(
            f"True={true_cls} | Pred={pred_cls}"
        )

        plt.legend()

        plt.savefig(
            os.path.join(
                save_dir,
                f"{i}_{true_cls}_to_{pred_cls}.png"
            )
        )

        plt.close()
