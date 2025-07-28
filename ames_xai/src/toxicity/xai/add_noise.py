import random
import pandas as pd

def add_smiles_char_noise(smiles):
    if len(smiles) < 2:
        return smiles  # Nothing to swap
    smiles = list(smiles)
    i, j = random.sample(range(len(smiles)), 2)
    smiles[i], smiles[j] = smiles[j], smiles[i]
    return ''.join(smiles)

smiles_path = "/home/dina/molprivacy/src/moreno/data_dir/test.csv"
df_test = pd.read_csv(smiles_path)
df_test_noisy = df_test.copy()
df_test_noisy['smiles'] = df_test_noisy['smiles'].apply(add_smiles_char_noise)
output_path = "/home/dina/molprivacy/src/moreno/data_dir/test_noisy.csv"

# Save noisy dataset
df_test_noisy.to_csv(output_path, index=False)
