import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from itertools import combinations
from pathlib import Path

def combine_all_gradcams(gradcam_dir, filter_sizes=None):
    gradcam_dir = Path(gradcam_dir)
    """
    Combine per-filter Grad-CAM arrays into a single token-level attribution.
    Expects files gradcam_filter_{i}.npy for i in filter_sizes.
    
    Returns:
        combined: np.ndarray of shape (n_rows, max_len)
    """
    if filter_sizes is None:
        filter_sizes = list(range(1, 21))

    # Load all existing filter files and collect shapes
    loaded = []
    for i, k in enumerate(filter_sizes, start=1):
        fp = gradcam_dir / f"gardcam_filter_{i}.npy"
        if not fp.exists():
            print(f"[WARN] Missing {fp.name}, skipping.")
            continue
        arr = np.load(fp)   # shape: (n_rows, L_k)
        loaded.append((k, arr))

    if not loaded:
        raise FileNotFoundError("No gradcam_filter_{i}.npy files found.")

    # Infer sizes
    n_rows = loaded[0][1].shape[0]
    # max_len = max over (L_k + k - 1)
    max_len = max(arr.shape[1] + k - 1 for (k, arr) in loaded)

    # Allocate accumulators
    combined = np.zeros((n_rows, max_len), dtype=np.float32)
    counts   = np.zeros((n_rows, max_len), dtype=np.float32)

    # Project each filter's window score to its tokens and accumulate
    for (k, cam) in loaded:
        L_k = cam.shape[1]
        # for each position j in the conv output, it covers tokens [j, j+k)
        for j in range(L_k):
            start = j
            end   = j + k
            # distribute equally to each token in window
            contrib = cam[:, j][:, None] / float(k)   # shape: (n_rows, 1)
            combined[:, start:end] += contrib
            counts[:,   start:end] += 1.0

    # Avoid division by zero; average overlapping contributions
    counts[counts == 0] = 1.0
    combined /= counts

    return combined  # shape: (n_rows, max_len)

def calculate_token_importance(attribution_values):
    if attribution_values.ndim == 2:
        return attribution_values.mean(axis=1)
    return attribution_values

def normalize(values):
    return (values - values.min()) / (values.max() + 1e-8 - values.min())

def compute_all_cosine_distances_with_gradcam(
    ig_values, shap_values, deeplift_values, occlusion_values, gradcam_values,
    smiles_list, label_matrix, group_names=None
):
    """
    Computes cosine distances for each (method pair, group),
    both for positive molecules and all molecules.
    Returns a DataFrame similar to your screenshot.
    """
    cosine_distances = {f"{a}_vs_{b}": {} for a, b in combinations(
        ["shap", "ig", "deeplift", "occlusion", "gradcam"], 2)}

    n_mols, n_groups = label_matrix.shape
    if group_names is None:
        group_names = [f"group_{i}" for i in range(n_groups)]

    # Initialize data containers
    groupwise_all = {k: {g: [] for g in range(n_groups)} for k in cosine_distances}
    groupwise_pos = {k: {g: [] for g in range(n_groups)} for k in cosine_distances}

    for molecule_index, molecule_smiles in enumerate(smiles_list):
        sequence_length = len(molecule_smiles)
        for group_index in range(n_groups):
            gradcam_index = molecule_index * n_groups + group_index

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

            # Compute cosine distances for all method pairs
            pairs = {
                "shap_vs_ig": cosine(shap_tokens, ig_tokens),
                "shap_vs_deeplift": cosine(shap_tokens, deeplift_tokens),
                "shap_vs_occlusion": cosine(shap_tokens, occlusion_tokens),
                "shap_vs_gradcam": cosine(shap_tokens, gradcam_tokens),
                "ig_vs_deeplift": cosine(ig_tokens, deeplift_tokens),
                "ig_vs_occlusion": cosine(ig_tokens, occlusion_tokens),
                "ig_vs_gradcam": cosine(ig_tokens, gradcam_tokens),
                "deeplift_vs_occlusion": cosine(deeplift_tokens, occlusion_tokens),
                "deeplift_vs_gradcam": cosine(deeplift_tokens, gradcam_tokens),
                "occlusion_vs_gradcam": cosine(occlusion_tokens, gradcam_tokens),
            }

            for pair, value in pairs.items():
                groupwise_all[pair][group_index].append(value)
                if label_matrix[molecule_index, group_index] == 1:
                    groupwise_pos[pair][group_index].append(value)

    # Build DataFrame like in your screenshot
    cols = []
    for g in group_names:
        cols += [f"{g}_positive", f"{g}_all"]
    cols += ["mean_positive", "mean_all"]

    df = pd.DataFrame(index=list(cosine_distances.keys()), columns=cols)

    for pair in cosine_distances.keys():
        mean_pos_list, mean_all_list = [], []
        for g_idx, g in enumerate(group_names):
            vals_all = np.array(groupwise_all[pair][g_idx])
            vals_pos = np.array(groupwise_pos[pair][g_idx])

            def fmt(x):
                return "NA" if x.size == 0 else f"{x.mean():.4f} ± {x.std():.4f}"

            df.at[pair, f"{g}_positive"] = fmt(vals_pos)
            df.at[pair, f"{g}_all"] = fmt(vals_all)
            if vals_pos.size:
                mean_pos_list.append(vals_pos.mean())
            if vals_all.size:
                mean_all_list.append(vals_all.mean())

        df.at[pair, "mean_positive"] = fmt(np.array(mean_pos_list))
        df.at[pair, "mean_all"] = fmt(np.array(mean_all_list))

    return df


def main():
    test_csv_path = "/home/dina/xai_methods/src/xai_benchmark/data_dir/test.csv"
    dataframe = pd.read_csv(test_csv_path)
    smiles_list = dataframe["smiles"].tolist()
    label_matrix = dataframe.iloc[:, 1:].values
    group_names = list(dataframe.columns[1:])

    ig_values = np.load("/home/dina/xai_methods/src/xai_benchmark/model/ig.npy")
    shap_values = np.load("/home/dina/xai_methods/src/xai_benchmark/model/shap.npy")
    deeplift_values = np.load("/home/dina/xai_methods/src/xai_benchmark/model/deeplift.npy")
    occlusion_values = np.load("/home/dina/xai_methods/src/xai_benchmark/model/occlusion.npy")
    #gradcam_values = np.load("/home/dina/xai_methods/src/xai_benchmark/model/gradcams/gardcam_filter_20.npy")

    gradcam_dir = "/home/dina/xai_methods/src/xai_benchmark/model/gradcams"
    gradcam_values = combine_all_gradcams(gradcam_dir)   # (n_mols * n_groups, max_len)

    np.save("/home/dina/xai_methods/src/xai_benchmark/model/combined_gradcam.npy", gradcam_values)

    df = compute_all_cosine_distances_with_gradcam(
        ig_values, shap_values, deeplift_values, occlusion_values, gradcam_values,
        smiles_list, label_matrix, group_names
    )

    # Save to CSV
    out_csv = "/home/dina/xai_methods/src/xai_benchmark/results/cosine_final.csv"
    df.to_csv(out_csv, index=True)
    print(f"Saved results to: {out_csv}")

    # Optionally, print a short preview
    with pd.option_context("display.max_columns", None, "display.width", 300):
        print(df)


if __name__ == "__main__":
    main()
