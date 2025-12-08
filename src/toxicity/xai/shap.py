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

    # Determine maximum sequence length for padding
    max_seq_len = max(batch[0].shape[1] for batch in test_loader)

    all_shap_values = []

    for batch in tqdm(test_loader, desc="Calculating SHAP"):
        inputs = batch[0].to(device)  # shape: (1, seq_len, embed_dim)
        batch_size, seq_len, embed_dim = inputs.shape

        for i in range(batch_size):
            single_input = inputs[i:i+1]

            # Pad input if needed
            if seq_len < max_seq_len:
                pad_size = max_seq_len - seq_len
                single_input = F.pad(single_input, (0, 0, 0, pad_size), value=0)

            shap_vals = explainer.attribute(
                single_input,
                baselines=torch.zeros_like(single_input),
                target=0,
                n_samples=num_samples
            )

            all_shap_values.append(shap_vals.detach().cpu().numpy())

            torch.cuda.empty_cache()

    final_shap = np.concatenate(all_shap_values, axis=0)  # (N, S, E)

    if save_path:
        np.save(save_path, final_shap)

    return final_shap