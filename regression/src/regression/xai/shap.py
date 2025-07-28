import torch
from captum.attr import GradientShap
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

def calculate_shap(model, test_loader, num_samples=100, save_path=None, device="cuda"):
    device = torch.device(device)
    model = model.to(device)
    model.eval()
    explainer = GradientShap(model)

    # Determine number of classes and max sequence length
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch[0].to(device)
            sample_output = model(inputs[:1])
            num_classes = sample_output.shape[1]
            break

    max_seq_len = max(batch[0].shape[1] for batch in test_loader)

    all_shap_values = []

    for batch in tqdm(test_loader, desc="Calculating SHAP"):
        inputs = batch[0].to(device)  # shape: (B, seq_len, embed_dim)
        batch_size, seq_len, embed_dim = inputs.shape

        # Pad the whole batch at once (more efficient)
        if seq_len < max_seq_len:
            pad_size = max_seq_len - seq_len
            inputs = F.pad(inputs, (0, 0, 0, pad_size), value=0)

        for i in range(batch_size):
            input_i = inputs[i:i+1]  # shape: (1, max_seq_len, embed_dim)
            mol_attributions = []

            for class_idx in range(num_classes):
                shap_vals = explainer.attribute(
                    input_i,
                    baselines=torch.zeros_like(input_i),
                    target=class_idx,
                    n_samples=num_samples
                )
                # Remove singleton batch dim to get shape: (seq_len, embed_dim)
                mol_attributions.append(shap_vals.squeeze(0).cpu().numpy())

            # Stack into shape: (C, S, E)
            mol_attributions = np.stack(mol_attributions, axis=0)
            all_shap_values.append(mol_attributions)

    final_shap = np.stack(all_shap_values, axis=0)  # (N, C, S, E)

    if save_path:
        np.save(save_path, final_shap)

    return final_shap
