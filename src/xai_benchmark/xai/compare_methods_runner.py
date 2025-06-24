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
    smiles_list, label_matrix
):
    cosine_distances = {
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

    for molecule_index, molecule_smiles in enumerate(smiles_list):
        sequence_length = len(molecule_smiles)
        functional_groups_present = np.where(label_matrix[molecule_index] == 1)[0]

        for group_index in functional_groups_present:
            try:
                gradcam_index = molecule_index * label_matrix.shape[1] + group_index

                ig_tokens = calculate_token_importance(ig_values[molecule_index, group_index])[1 : sequence_length + 1]
                shap_tokens = calculate_token_importance(shap_values[molecule_index, group_index])[1 : sequence_length + 1]
                deeplift_tokens = calculate_token_importance(deeplift_values[molecule_index, group_index])[1 : sequence_length + 1]
                occlusion_tokens = calculate_token_importance(occlusion_values[molecule_index, group_index])[1 : sequence_length + 1]
                gradcam_tokens = gradcam_values[gradcam_index][1 : sequence_length + 1]

                ig_tokens = normalize(ig_tokens)
                shap_tokens = normalize(shap_tokens)
                deeplift_tokens = normalize(deeplift_tokens)
                occlusion_tokens = normalize(occlusion_tokens)
                gradcam_tokens = normalize(gradcam_tokens)

                cosine_distances["shap_vs_ig"].append(cosine(shap_tokens, ig_tokens))
                cosine_distances["shap_vs_deeplift"].append(cosine(shap_tokens, deeplift_tokens))
                cosine_distances["shap_vs_occlusion"].append(cosine(shap_tokens, occlusion_tokens))
                cosine_distances["shap_vs_gradcam"].append(cosine(shap_tokens, gradcam_tokens))
                cosine_distances["ig_vs_deeplift"].append(cosine(ig_tokens, deeplift_tokens))
                cosine_distances["ig_vs_occlusion"].append(cosine(ig_tokens, occlusion_tokens))
                cosine_distances["ig_vs_gradcam"].append(cosine(ig_tokens, gradcam_tokens))
                cosine_distances["deeplift_vs_occlusion"].append(cosine(deeplift_tokens, occlusion_tokens))
                cosine_distances["deeplift_vs_gradcam"].append(cosine(deeplift_tokens, gradcam_tokens))
                cosine_distances["occlusion_vs_gradcam"].append(cosine(occlusion_tokens, gradcam_tokens))

            except Exception as error:
                print(f"Skipping molecule {molecule_index}, group {group_index} due to error: {error}")

    return cosine_distances

def main():
    test_csv_path = "/home/dina/xai_methods/src/xai_benchmark/data_dir/test.csv"
    dataframe = pd.read_csv(test_csv_path)
    smiles_list = dataframe["smiles"].tolist()
    label_matrix = dataframe.iloc[:, 1:].values

    ig_values = np.load("/home/dina/xai_methods/src/xai_benchmark/model/ig.npy")
    shap_values = np.load("/home/dina/xai_methods/src/xai_benchmark/model/shap.npy")
    deeplift_values = np.load("/home/dina/xai_methods/src/xai_benchmark/model/deeplift.npy")
    occlusion_values = np.load("/home/dina/xai_methods/src/xai_benchmark/model/occlusion.npy")
    gradcam_values = np.load("/home/dina/xai_methods/src/xai_benchmark/model/gradcams/gardcam_filter_20.npy")

    results = compute_all_cosine_distances_with_gradcam(
        ig_values, shap_values, deeplift_values, occlusion_values,
        gradcam_values, smiles_list, label_matrix
    )

    for method_pair, cosine_scores in results.items():
        mean_score = np.mean(cosine_scores)
        std_dev = np.std(cosine_scores)
        print(f"{method_pair}: {mean_score:.4f} ± {std_dev:.4f}")

if __name__ == "__main__":
    main()
