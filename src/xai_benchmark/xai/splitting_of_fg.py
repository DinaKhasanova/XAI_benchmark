import os
import pandas as pd


source_dir = "../data/separate_df_clean"
output_dir = ".."
alcohol_dir = "../data_dir"

# Load the split files
train_smiles = set(pd.read_csv(os.path.join(alcohol_dir, "train.csv"))["smiles"])
val_smiles = set(pd.read_csv(os.path.join(alcohol_dir, "validation.csv"))["smiles"])
test_smiles = set(pd.read_csv(os.path.join(alcohol_dir, "test.csv"))["smiles"])

# Process other functional groups
for file in os.listdir(source_dir):
    if file.startswith("clean_") and file.endswith(".csv"):
        functional_group = file.replace("clean_", "").replace(".csv", "")

        if functional_group == "alcohol":
            continue  # Skip alcohol

        # Create directory inside the base output path
        data_dir = os.path.join(output_dir, f"data_dir_{functional_group}")
        os.makedirs(data_dir, exist_ok=True)

        # Load dataset
        df = pd.read_csv(os.path.join(source_dir, file))

        # Assign the same splits as alcohol
        df_train = df[df["smiles"].isin(train_smiles)]
        df_val = df[df["smiles"].isin(val_smiles)]
        df_test = df[df["smiles"].isin(test_smiles)]

        # Save the splits
        df_train.to_csv(os.path.join(data_dir, "train.csv"), index=False)
        df_val.to_csv(os.path.join(data_dir, "validation.csv"), index=False)
        df_test.to_csv(os.path.join(data_dir, "test.csv"), index=False)
