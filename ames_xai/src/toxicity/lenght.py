import numpy as np
import pandas as pd
from moreno_encoders.utils.encodings import generate_matrix_encodings

#code just to double check the shape of explanations and test dataset
ig = np.load("model_captum_1/ig.npy")  # Shape: (1605, 198, 64)
test_df = pd.read_csv("/home/dina/molprivacy/src/moreno/data_dir/test.csv")

smiles = test_df["smiles"]

encoding_list = generate_matrix_encodings(smiles)

# Create a DataFrame with SMILES, their lengths and encoding lengths
df_lengths = pd.DataFrame({
    "smiles": smiles,
    "smiles_length": smiles.apply(len),
    "encoding_length": [len(encoding) for encoding in encoding_list]
})
df_lengths["difference"] = abs(df_lengths["smiles_length"] - df_lengths["encoding_length"])
df_f = df_lengths[df_lengths["difference"] > 2] #SOS and EOS tokens
df_f.to_csv("df.csv", index=False) 
