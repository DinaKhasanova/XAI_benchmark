import torch
from captum.attr import GradientShap
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

def calculate_shap(model, test_loader, num_samples=100, save_path=None, device="cuda"):
    device = torch.device(device)
    model = model.to(device)
    explainer = GradientShap(model)

    # Determine number of classes from model output
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch[0].to(device)
            sample_output = model(inputs[:1])
            num_classes = sample_output.shape[1]
            break

    # Determine maximum sequence length for padding
    max_seq_len = max(batch[0].shape[1] for batch in test_loader)

    all_shap_values = []

    for batch in tqdm(test_loader, desc="Calculating SHAP"):
        inputs = batch[0].to(device)
        batch_size, seq_len, embed_dim = inputs.shape

        # Pad input if needed
        if seq_len < max_seq_len:
            pad_size = max_seq_len - seq_len
            inputs = F.pad(inputs, (0, 0, 0, pad_size), value=0)

        batch_attributions = []

        for class_idx in range(num_classes):
            shap_vals = explainer.attribute(
                inputs,
                baselines=torch.zeros_like(inputs),
                target=class_idx,
                n_samples=num_samples
            )
            batch_attributions.append(shap_vals.detach().cpu().numpy())

            torch.cuda.empty_cache()

        # Stack SHAP values for each class: (num_classes, batch_size, seq_len, embed_dim)
        batch_attributions = np.stack(batch_attributions, axis=1)  # (B, C, S, E)
        all_shap_values.append(batch_attributions)

    final_shap = np.concatenate(all_shap_values, axis=0)  # (N, C, S, E)

    if save_path:
        np.save(save_path, final_shap)

    return final_shap
