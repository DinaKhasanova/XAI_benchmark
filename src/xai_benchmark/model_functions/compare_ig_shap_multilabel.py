
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine

def compare_ig_shap_multilabel(ig_path: str, shap_path: str, label_csv_path: str):
    ig = np.load(ig_path)
    shap = np.load(shap_path)

    df = pd.read_csv(label_csv_path)
    labels = df.drop(columns=["smiles"]).values  # shape: (num_samples, num_classes)

    if ig.shape != shap.shape:
        raise ValueError(f"Shape mismatch: IG shape {ig.shape} vs SHAP shape {shap.shape}")

    if labels.shape[0] != ig.shape[0] or labels.shape[1] != ig.shape[1]:
        raise ValueError(f"Label shape {labels.shape} doesn't match attribution shape {ig.shape[:2]}")

    cosine_distances = []

    for i in range(ig.shape[0]):  # samples
        for c in range(ig.shape[1]):  # classes
            if labels[i, c] != 1:
                continue
            flat_ig = ig[i, c].flatten()
            flat_shap = shap[i, c].flatten()
            cos_dist = cosine(flat_ig, flat_shap)
            cosine_distances.append(cos_dist)

    if not cosine_distances:
        raise ValueError("No valid comparisons: no class with label=1 was found.")

    mean_cosine_distance = np.mean(cosine_distances)
    print(f"Mean cosine distance between IG and SHAP (label=1 only): {mean_cosine_distance:.4f}")
    return mean_cosine_distance


# Example usage
if __name__ == "__main__":
    ig_path = "ig.npy"
    shap_path = "shap.npy"
    label_csv_path = "/home/dina/xai_methods/src/xai_benchmark/data_dir/test.csv"
    compare_ig_shap_multilabel(ig_path, shap_path, label_csv_path)
