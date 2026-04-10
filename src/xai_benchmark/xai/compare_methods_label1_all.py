
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
import pickle

def calculate_token_importance(attribution):
    if attribution.ndim == 2:
        return attribution.mean(axis=1)
    return attribution

def extract_token_importance_for_smiles(token_importances, smiles):
    return token_importances[:len(smiles)]

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
    # Replace with actual paths
    test_csv = "../data_dir/test.csv"
    smiles = pd.read_csv(test_csv)["smiles"].tolist()

    ig = np.load("../model/ig.npy")
    import pickle
    shap_path = "../model/shap.npy"
    shap_data = np.load(shap_path, allow_pickle=True)
    pickled_data = shap_data["shap/data.pkl"]
    shap = pickle.loads(pickled_data)
    shap = shap.reshape(shap.shape[0], shap.shape[1], shap.shape[2])
    deeplift = np.load("../model/deeplift.npy")
    occlusion = np.load("../model/occlusion.npy")
    gradcam = np.load("../model/gradcam_token_attributions.npy")

    results = compute_all_cosine_distances_with_gradcam(ig, shap, deeplift, occlusion, gradcam, smiles)

    for key, values in results.items():
        mean = np.mean(values)
        std = np.std(values)
        print(f"{key}: {mean:.4f} ± {std:.4f}")
