import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine

def calculate_token_importance(attribution_values):
    if attribution_values.ndim == 2:
        return attribution_values.mean(axis=1)
    return attribution_values

def normalize(values):
    return (values - values.min()) / (values.max() + 1e-8 - values.min())

def compute_all_cosine_distances_with_gradcam(
    ig_values, shap_values, deeplift_values, occlusion_values, gradcam_values,
    smiles_list, label_matrix, filter_positive_only
):
    method_pairs = [
        ("shap", "ig"), ("shap", "deeplift"), ("shap", "occlusion"), ("shap", "gradcam"),
        ("ig", "deeplift"), ("ig", "occlusion"), ("ig", "gradcam"),
        ("deeplift", "occlusion"), ("deeplift", "gradcam"),
        ("occlusion", "gradcam")
    ]
    per_group_cosine_distances = {pair: {g: [] for g in range(label_matrix.shape[1])} for pair in [f"{a}_vs_{b}" for a, b in method_pairs]}
    for molecule_index, molecule_smiles in enumerate(smiles_list):
        sequence_length = len(molecule_smiles)
        if filter_positive_only:
            group_indices = np.where(label_matrix[molecule_index] == 1)[0]
        else:
            group_indices = range(label_matrix.shape[1])
        for group_index in group_indices:
            gradcam_index = molecule_index * label_matrix.shape[1] + group_index
            tokens_dict = {
                "ig": normalize(calculate_token_importance(ig_values[molecule_index, group_index])[1:sequence_length+1]),
                "shap": normalize(calculate_token_importance(shap_values[molecule_index, group_index])[1:sequence_length+1]),
                "deeplift": normalize(calculate_token_importance(deeplift_values[molecule_index, group_index])[1:sequence_length+1]),
                "occlusion": normalize(calculate_token_importance(occlusion_values[molecule_index, group_index])[1:sequence_length+1]),
                "gradcam": normalize(gradcam_values[gradcam_index][1:sequence_length+1])
            }
            for a, b in method_pairs:
                pair_name = f"{a}_vs_{b}"
                dist = cosine(tokens_dict[a], tokens_dict[b])
                per_group_cosine_distances[pair_name][group_index].append(dist)
    summary = {}
    for pair in per_group_cosine_distances:
        summary[pair] = {
            g: (
                f"{np.mean(per_group_cosine_distances[pair][g]):.4f} ± {np.std(per_group_cosine_distances[pair][g]):.4f}"
                if per_group_cosine_distances[pair][g] else np.nan
            ) for g in per_group_cosine_distances[pair]
        }
        # Add mean across all functional groups
        all_values = [d for group in per_group_cosine_distances[pair].values() for d in group]
        if all_values:
            mean_val = np.mean(all_values)
            std_val = np.std(all_values)
            summary[pair]["mean_all_groups"] = f"{mean_val:.4f} ± {std_val:.4f}"
        else:
            summary[pair]["mean_all_groups"] = np.nan
    return summary

def main():
    test_csv_path = "/home/dina/regression/src/regression/data_dir/test.csv"
    dataframe = pd.read_csv(test_csv_path)
    smiles_list = dataframe["smiles"].tolist()
    label_matrix = dataframe.iloc[:, 1:].values
    group_names = {i: name for i, name in enumerate(dataframe.columns[1:])}
    ig_values = np.load("/home/dina/regression/src/regression/model/ig.npy")
    shap_values = np.load("/home/dina/regression/src/regression/model/shap.npy")
    deeplift_values = np.load("/home/dina/regression/src/regression/model/deeplift.npy")
    occlusion_values = np.load("/home/dina/regression/src/regression/model/occlusion.npy")
    gradcam_values = np.load("/home/dina/regression/src/regression/model/gradcams/gardcam_filter_1.npy")

    positive_summary = compute_all_cosine_distances_with_gradcam(
        ig_values, shap_values, deeplift_values, occlusion_values,
        gradcam_values, smiles_list, label_matrix, filter_positive_only=True
    )
    all_summary = compute_all_cosine_distances_with_gradcam(
        ig_values, shap_values, deeplift_values, occlusion_values,
        gradcam_values, smiles_list, label_matrix, filter_positive_only=False
    )

    df_list = []
    for pair in positive_summary:
        row = {"method_pair": pair}
        for g in group_names:
            row[f"{group_names[g]}_positive"] = positive_summary[pair].get(g, np.nan)
            row[f"{group_names[g]}_all"] = all_summary[pair].get(g, np.nan)
        row["mean_positive"] = positive_summary[pair].get("mean_all_groups", np.nan)
        row["mean_all"] = all_summary[pair].get("mean_all_groups", np.nan)
        df_list.append(row)

    df_final = pd.DataFrame(df_list)
    df_final.to_csv("/home/dina/regression/src/regression/model/cosine_distance_summary.csv", index=False)

if __name__ == "__main__":
    main()
