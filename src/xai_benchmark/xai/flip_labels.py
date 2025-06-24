import random
import pandas as pd

def flip_labels(df, label_column='label', flip_fraction=0.2, seed=42):
    random.seed(seed)
    all_indices = df.index.tolist()
    num_to_flip = int(len(all_indices) * flip_fraction)
    indices_to_flip = random.sample(all_indices, num_to_flip)

    df_flipped = df.copy()
    df_flipped.loc[indices_to_flip, label_column] = df_flipped.loc[indices_to_flip, label_column].apply(lambda x: 1 - x)

    return df_flipped

smiles_path = "../data_dir/test.csv"
output_path = "../data_dir/test_label.csv"

df_test = pd.read_csv(smiles_path)

df_test_label = flip_labels(df_test)

df_test_label.to_csv(output_path, index=False)
