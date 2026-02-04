import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from utils import import_dataset_mclab

# ============================================================
# Global plotting style
# ============================================================
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 14,
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 13,
    "axes.linewidth": 1.2,
    "grid.alpha": 0.3,
    "grid.linestyle": "--"
})

# ============================================================
# Models to analyze
# ============================================================
MODELS = {
    "Random Forest": "../Random_Forest/mclab_forest.sav",
    "XGBoost": "../XGBOOST/mclab_xgboost.sav",
}

FEATURE_COLUMNS = ['distance', 'elevations', 'antenna_height']

OUTPUT_DIR = "feature_importance_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# Load dataset
# ============================================================
X_train, X_test, X_val, y_train, y_test, y_val, scaler = import_dataset_mclab()

# ============================================================
# Create figure with subplots
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

for ax, (model_name, model_path) in zip(axes, MODELS.items()):

    model = joblib.load(model_path)

    if not hasattr(model, "feature_importances_"):
        raise RuntimeError(f"{model_name} does not provide native feature importance.")

    importances = model.feature_importances_

    # Normalize for visualization only
    importances = importances / np.sum(importances)

    order = np.argsort(importances)

    ax.barh(
        np.array(FEATURE_COLUMNS)[order],
        importances[order],
        color="dodgerblue"
    )

    ax.set_title(model_name)
    ax.set_xlabel("Normalized Importance")
    ax.grid(axis="x")

axes[0].set_ylabel("Feature")

plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "fig5_feature_importance_rf_xgboost.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
