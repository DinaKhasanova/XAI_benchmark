import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine

def compare_ig_shap_per_class(ig_path: str, shap_path: str, label_csv_path: str):
    ig = np.load(ig_path)
    shap = np.load(shap_path)

    df = pd.read_csv(label_csv_path)
    class_names = df.columns.drop("smiles").tolist()
    labels = df[class_names].values  # shape: (num_samples, num_classes)

    if ig.shape != shap.shape:
        raise ValueError(f"Shape mismatch: IG shape {ig.shape} vs SHAP shape {shap.shape}")
    if labels.shape[0] != ig.shape[0] or labels.shape[1] != ig.shape[1]:
        raise ValueError(f"Label shape {labels.shape} doesn't match attribution shape {ig.shape[:2]}")

    class_cosine_dists = {name: [] for name in class_names}

    for i in range(ig.shape[0]):  # samples
        for c, name in enumerate(class_names):  # classes
            if labels[i, c] != 1:
                continue
            flat_ig = ig[i, c].flatten()
            flat_shap = shap[i, c].flatten()
            cos_dist = cosine(flat_ig, flat_shap)
            class_cosine_dists[name].append(cos_dist)

    # Compute mean per class
    mean_per_class = {
        name: np.mean(dists) if dists else np.nan
        for name, dists in class_cosine_dists.items()
    }

    # Print results
    print("Mean cosine distance between IG and SHAP per class (label=1 only):")
    for name, value in mean_per_class.items():
        print(f"{name}: {value:.4f}" if not np.isnan(value) else f"{name}: No label=1 samples")

    return mean_per_class


# Example usage
if __name__ == "__main__":
    ig_path = "ig.npy"
    shap_path = "shap.npy"
    label_csv_path = "/home/dina/xai_methods/src/xai_benchmark/data_dir/test.csv"
    compare_ig_shap_per_class(ig_path, shap_path, label_csv_path)
