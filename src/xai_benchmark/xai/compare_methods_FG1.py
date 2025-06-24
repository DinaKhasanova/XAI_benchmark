import numpy as np
import pickle
import pandas as pd
from scipy.spatial.distance import cosine
from rdkit import Chem

smart = Chem.MolFromSmarts("[NX3;H2,H1;!$(NC=O)]")

def load_data(ig_path, shap_path, deeplift_path, occlusion_path, smiles_path):
    ig = np.load(ig_path)  # shape: (N, T, D)
    shap_data = np.load(shap_path, allow_pickle=True)
    deeplift = np.load(deeplift_path)
    occlusion = np.load(occlusion_path)
    pickled_data = shap_data['shap/data.pkl']
    shap = pickle.loads(pickled_data)  # shape: (N, T, D)
    shap = shap.reshape(shap.shape[0], shap.shape[1], shap.shape[2])
    test_df = pd.read_csv(smiles_path)
    label_one = test_df[test_df["label"]==1].reset_index(drop=True)
    smiles = label_one["smiles"].tolist()
    mask = test_df["label"] == 1
    ig = ig[mask.to_numpy()]
    shap = shap[mask.to_numpy()]
    deeplift = deeplift[mask.to_numpy()]
    occlusion = occlusion[mask.to_numpy()]
    return ig, shap, deeplift, occlusion, smiles

def calculate_token_importance(values):
    non_zero_mask = ~np.all(values == 0, axis=1)
    filtered_values = values[non_zero_mask]
    token_importance = np.sum(filtered_values, axis=1)  # sum across embedding dim
    return token_importance

def extract_token_importance_for_smiles(token_importance, smiles_string):
    # Removes [CLS] and [PAD] tokens, keeps only SMILES tokens
    return token_importance[1:len(smiles_string)+1]

def compute_all_cosine_distances(ig, shap, deeplift, occlusion, smiles):
    distances = {
        "shap_vs_ig": [],
        "shap_vs_deeplift": [],
        "shap_vs_occlusion": [],
        "ig_vs_deeplift": [],
        "ig_vs_occlusion": [],
        "deeplift_vs_occlusion": []
    }

    for i in range(len(smiles)):
        ig_values = ig[i]      # shape: (T, D)
        shap_values = shap[i]  # shape: (T, D)
        deeplift_values = deeplift[i]  # shape: (T, D)
        occlusion_values = occlusion[i]
        # Reduce to per-token importance: shape → [T]
        ig_importance = calculate_token_importance(ig_values)
        shap_importance = calculate_token_importance(shap_values)
        deeplift_importance = calculate_token_importance(deeplift_values)
        occlusion_importance = calculate_token_importance(occlusion_values)
        # Trim padding based on SMILES length
        ig_trimmed = extract_token_importance_for_smiles(ig_importance, smiles[i])
        shap_trimmed = extract_token_importance_for_smiles(shap_importance, smiles[i])
        deeplift_trimmed = extract_token_importance_for_smiles(deeplift_importance, smiles[i])
        occlusion_trimmed = extract_token_importance_for_smiles(occlusion_importance, smiles[i])
        # Ensure matching lengths
        if not (len(ig_trimmed) == len(shap_trimmed) == len(deeplift_trimmed) == len(occlusion_trimmed)):
            print(f"Skipping molecule {i} due to length mismatch: IG={len(ig_trimmed)}, SHAP={len(shap_trimmed)}, DEEPLIFT={len(deeplift_trimmed)}, OCCLUSION={len(occlusion_trimmed)}")
            continue

        # Compute cosine distances
        distances["shap_vs_ig"].append(cosine(shap_trimmed, ig_trimmed))
        distances["shap_vs_deeplift"].append(cosine(shap_trimmed, deeplift_trimmed))
        distances["ig_vs_deeplift"].append(cosine(ig_trimmed, deeplift_trimmed))
        distances["shap_vs_occlusion"].append(cosine(shap_trimmed, occlusion_trimmed))
        distances["ig_vs_occlusion"].append(cosine(ig_trimmed, occlusion_trimmed))
        distances["deeplift_vs_occlusion"].append(cosine(deeplift_trimmed, occlusion_trimmed))
    return distances

def one_fg(ig, shap, deeplift, occlusion, smiles):
    valid_ig, valid_shap, valid_deeplift, valid_occlusion, valid_smiles = [], [], [], [], []

    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        matches = mol.GetSubstructMatches(smart)
        if len(matches) == 1:
            valid_ig.append(ig[i])
            valid_shap.append(shap[i])
            valid_deeplift.append(deeplift[i])
            valid_occlusion.append(occlusion[i])
            valid_smiles.append(smi)

    print(f"Molecules with exactly one functional group: {len(valid_smiles)}")
    return (
        np.array(valid_ig),
        np.array(valid_shap),
        np.array(valid_deeplift),
        np.array(valid_occlusion),
        valid_smiles
    )

def main():
    ig_path = "/home/dina/molprivacy/src/moreno/model/ig.npy"
    shap_path = "/home/dina/molprivacy/src/moreno/model/shap.npy"
    smiles_path = "/home/dina/molprivacy/src/moreno/data_dir/test.csv"
    deeplift_path = "/home/dina/molprivacy/src/moreno/model/deeplift.npy"
    occlusion_path = "/home/dina/molprivacy/src/moreno/model/occlusion.npy"
    ig, shap, deeplift, occlusion, smiles = load_data(ig_path, shap_path, deeplift_path, occlusion_path, smiles_path)
    ig, shap, deeplift, occlusion, smiles = one_fg(ig, shap, deeplift, occlusion, smiles)
    distances = compute_all_cosine_distances(ig, shap, deeplift, occlusion, smiles)

    print("Mean cosine distances between attribution methods:\n")
    for key in distances:
        if len(distances[key]) > 0:
            print(f"{key}: {np.mean(distances[key]):.4f} ± {np.std(distances[key]):.4f}")
        else:
            print(f"{key}: no valid comparisons (skipped due to length mismatch)")

if __name__ == "__main__":
    main()

