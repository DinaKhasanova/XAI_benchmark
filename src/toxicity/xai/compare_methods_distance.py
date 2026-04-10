
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine

from pathlib import Path
import numpy as np
import os
def combine_all_gradcams(gradcam_dir, filter_sizes=None):
    """
    Combine per-filter Grad-CAM arrays into a single token-level attribution using the window approach.
    Expects files named 'gradcam_filter_{i}.npy' inside gradcam_dir for i in filter_sizes.
    Returns: np.ndarray of shape (n_rows, max_len) with averaged token-level scores.
    """
    gradcam_dir = Path(gradcam_dir)
    if filter_sizes is None:
        filter_sizes = list(range(1, 21))  # default filters 1..20

    # Load all existing filter files
    loaded = []
    for i, k in enumerate(filter_sizes, start=1):
        fp = gradcam_dir / f"gradcam_filter_{i}.npy"
        if not fp.exists():
            continue
        cam = np.load(fp)  # shape: (n_rows, L_k)
        loaded.append((k, cam))

    if not loaded:
        raise FileNotFoundError(f"No Grad-CAM filter files found in {gradcam_dir}")

    n_rows = loaded[0][1].shape[0]
    max_len = max(arr.shape[1] + k - 1 for (k, arr) in loaded)

    combined = np.zeros((n_rows, max_len), dtype=np.float32)
    counts   = np.zeros((n_rows, max_len), dtype=np.float32)

    for (k, cam) in loaded:
        L_k = cam.shape[1]
        for j in range(L_k):
            start = j
            end   = j + k
            contrib = cam[:, j][:, None] / float(k)
            combined[:, start:end] += contrib
            counts[:,   start:end] += 1.0

    counts[counts == 0] = 1.0
    combined /= counts
    return combined



def calculate_token_importance(attribution):
    if attribution.ndim == 2:
        return attribution.mean(axis=1)
    return attribution

def normalize(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)

def compute_all_cosine_distances_with_gradcam(ig, shap, deeplift, occlusion, gradcam, smiles):
    distances = {
        "shap_vs_ig": [],
        "shap_vs_deeplift": [],
        "shap_vs_occlusion": [],
        "shap_vs_gradcam": [],
        "ig_vs_deeplift": [],
        "ig_vs_occlusion": [],
        "ig_vs_gradcam": [],
        "deeplift_vs_occlusion": [],
        "deeplift_vs_gradcam": [],
        "occlusion_vs_gradcam": []
    }

    for i, smi in enumerate(smiles):
        try:
            smiles_len = len(smi)

            ig_imp = calculate_token_importance(ig[i])[1 : smiles_len + 1]
            shap_imp = calculate_token_importance(shap[i])[1 : smiles_len + 1]
            deeplift_imp = calculate_token_importance(deeplift[i])[1 : smiles_len + 1]
            occlusion_imp = calculate_token_importance(occlusion[i])[1 : smiles_len + 1]
            gradcam_imp = gradcam[i][:smiles_len]

            ig_trim = normalize(ig_imp)
            shap_trim = normalize(shap_imp)
            deeplift_trim = normalize(deeplift_imp)
            occlusion_trim = normalize(occlusion_imp)
            gradcam_trim = normalize(gradcam_imp)

            all_lengths = {len(ig_trim), len(shap_trim), len(deeplift_trim), len(occlusion_trim), len(gradcam_trim)}
            if len(all_lengths) != 1:
                print(f"Skipping index {i} due to length mismatch: {all_lengths}")
                continue

            distances["shap_vs_ig"].append(cosine(shap_trim, ig_trim))
            distances["shap_vs_deeplift"].append(cosine(shap_trim, deeplift_trim))
            distances["shap_vs_occlusion"].append(cosine(shap_trim, occlusion_trim))
            distances["shap_vs_gradcam"].append(cosine(shap_trim, gradcam_trim))
            distances["ig_vs_deeplift"].append(cosine(ig_trim, deeplift_trim))
            distances["ig_vs_occlusion"].append(cosine(ig_trim, occlusion_trim))
            distances["ig_vs_gradcam"].append(cosine(ig_trim, gradcam_trim))
            distances["deeplift_vs_occlusion"].append(cosine(deeplift_trim, occlusion_trim))
            distances["deeplift_vs_gradcam"].append(cosine(deeplift_trim, gradcam_trim))
            distances["occlusion_vs_gradcam"].append(cosine(occlusion_trim, gradcam_trim))

        except Exception as e:
            print(f"Error at index {i}: {e}")
            continue

    return distances

if __name__ == "__main__":
    df = pd.read_csv("../data_dir/test.csv")
    smiles = df["smiles"].tolist()
    toxicity = df["label"].values

    ig = np.load("../model/ig.npy")
    shap = np.load("../model/shap.npy")
    deeplift = np.load("../model/deeplift.npy")
    occlusion = np.load("../model/occlusion.npy")
    gradcam = combine_all_gradcams("../model/gradcams", filter_sizes=list(range(1,21)))
    combined_gradcam_path = "../model/gradcam_combined.npy"
    os.makedirs(os.path.dirname(combined_gradcam_path), exist_ok=True)
    np.save(combined_gradcam_path, gradcam)
    print(f"Saved combined Grad-CAM to {combined_gradcam_path} with shape {gradcam.shape}")
    mask_positive = toxicity == 1

    results_all = compute_all_cosine_distances_with_gradcam(
        ig, shap, deeplift, occlusion, gradcam, smiles
    )

    results_pos = compute_all_cosine_distances_with_gradcam(
        ig[mask_positive], shap[mask_positive], deeplift[mask_positive],
        occlusion[mask_positive], gradcam[mask_positive],
        list(np.array(smiles)[mask_positive])
    )

    summary = []
    for method in results_all.keys():
        mean_pos = np.mean(results_pos[method])
        std_pos = np.std(results_pos[method])
        mean_all = np.mean(results_all[method])
        std_all = np.std(results_all[method])

        row = {
            "method": method,
            "positive_1": f"{mean_pos:.4f} ± {std_pos:.4f}",
            "all_labels": f"{mean_all:.4f} ± {std_all:.4f}"
        }
        summary.append(row)

    pd.DataFrame(summary).to_csv("../model/xai_distances_summary_all.csv", index=False)
    print("Saved to xai_distances_summary_all.csv")
