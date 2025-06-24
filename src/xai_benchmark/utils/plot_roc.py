from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def plot_roc(y_preds: np.ndarray, y_trues: np.ndarray, save_location: Path):
    """
    Plot micro-averaged ROC curve for multi-label classification.

    Parameters:
    y_preds: np.ndarray - predicted probabilities, shape (n_samples, n_classes)
    y_trues: np.ndarray - binary true labels, shape (n_samples, n_classes)
    """
    # Flatten all class predictions
    fpr, tpr, _ = roc_curve(y_trues.ravel(), y_preds.ravel())
    roc_auc = auc(fpr, tpr)

    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 6))
    sns.lineplot(x=fpr, y=tpr, label=f"Micro-Avg ROC (AUC = {roc_auc:.2f})", lw=2)
    sns.lineplot(x=[0, 1], y=[0, 1], linestyle="--", color="gray", lw=2)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Micro-Averaged ROC Curve for Multi-Label Classification")
    plt.legend(loc="lower right")

    save_location.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_location / "micro_avg_roc.pdf", format="pdf")
    plt.close()
