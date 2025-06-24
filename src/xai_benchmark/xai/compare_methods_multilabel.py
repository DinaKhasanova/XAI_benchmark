
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

def compute_all_cosine_distances_with_gradcam(ig, shap, deeplift, occlusion, gradcam, smiles, num_classes):
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
        smiles_len = len(smi)
        for j in range(num_classes):
            try:
                idx = i * num_classes + j  # Grad-CAM is flattened over samples and classes

                ig_imp = calculate_token_importance(ig[i, j])
                shap_imp = calculate_token_importance(shap[i, j])
                deeplift_imp = calculate_token_importance(deeplift[i, j])
                occlusion_imp = calculate_token_importance(occlusion[i, j])
                gradcam_imp = gradcam[idx]

                ig_imp = normalize(extract_token_importance_for_smiles(ig_imp, smi))
                shap_imp = normalize(extract_token_importance_for_smiles(shap_imp, smi))
                deeplift_imp = normalize(extract_token_importance_for_smiles(deeplift_imp, smi))
                occlusion_imp = normalize(extract_token_importance_for_smiles(occlusion_imp, smi))
                gradcam_imp = normalize(extract_token_importance_for_smiles(gradcam_imp, smi))

                distances["shap_vs_ig"].append(1 - cosine(shap_imp, ig_imp))
                distances["shap_vs_deeplift"].append(1 - cosine(shap_imp, deeplift_imp))
                distances["shap_vs_occlusion"].append(1 - cosine(shap_imp, occlusion_imp))
                distances["shap_vs_gradcam"].append(1 - cosine(shap_imp, gradcam_imp))
                distances["ig_vs_deeplift"].append(1 - cosine(ig_imp, deeplift_imp))
                distances["ig_vs_occlusion"].append(1 - cosine(ig_imp, occlusion_imp))
                distances["ig_vs_gradcam"].append(1 - cosine(ig_imp, gradcam_imp))
                distances["deeplift_vs_occlusion"].append(1 - cosine(deeplift_imp, occlusion_imp))
                distances["deeplift_vs_gradcam"].append(1 - cosine(deeplift_imp, gradcam_imp))
                distances["occlusion_vs_gradcam"].append(1 - cosine(occlusion_imp, gradcam_imp))

            except Exception as e:
                print(f"Skipping sample {i}, class {j} due to error: {e}")

    return distances
