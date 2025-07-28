
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine

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
    df = pd.read_csv("/home/dina/ames_xai/src/toxicity/data_dir/test.csv")
    smiles = df["smiles"].tolist()
    toxicity = df["label"].values

    ig = np.load("/home/dina/ames_xai/src/toxicity/model/ig.npy")
    shap = np.load("/home/dina/ames_xai/src/toxicity/model/shap.npy")
    deeplift = np.load("/home/dina/ames_xai/src/toxicity/model/deeplift.npy")
    occlusion = np.load("/home/dina/ames_xai/src/toxicity/model/occlusion.npy")
    gradcam = np.load("/home/dina/ames_xai/src/toxicity/model/gradcams/gardcam_filter_1.npy")

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

    pd.DataFrame(summary).to_csv("/home/dina/ames_xai/src/toxicity/model/xai_distances_summary.csv", index=False)
    print("Saved to xai_distances_summary.csv")
