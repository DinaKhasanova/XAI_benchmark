from captum.attr import IntegratedGradients
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

def calculate_ig(model, test_loader, device="cuda", save_path=None):
    device = torch.device(device)
    model = model.to(device)
    ig = IntegratedGradients(model)
    ig_results = []

    # Determine number of classes from model output
    with torch.no_grad():
        for x, _ in test_loader:
            sample_output = model(x.to(device))
            num_classes = sample_output.shape[1]
            break

    # Determine the maximum sequence length in the dataset for padding
    max_sequence_length = max(batch[0].size(1) for batch in test_loader)

    for batch_idx, batch in enumerate(tqdm(test_loader, desc="Calculating IG")):
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        inputs.requires_grad_()

        batch_attributions = []

        for class_idx in range(num_classes):
            attr = ig.attribute(
                inputs=inputs,
                baselines=torch.zeros_like(inputs),
                target=class_idx,
                internal_batch_size=1
            )

            # Pad attributions to max_sequence_length
            seq_length = attr.size(1)
            if seq_length < max_sequence_length:
                padding_needed = max_sequence_length - seq_length
                attr = torch.cat(
                    [attr, torch.zeros(attr.size(0), padding_needed, *attr.shape[2:], device=device)],
                    dim=1
                )

            batch_attributions.append(attr.detach().cpu().numpy())

        batch_attributions = np.stack(batch_attributions, axis=1)  # (batch_size, num_classes, ...)
        ig_results.append(batch_attributions)

    final_attr = np.concatenate(ig_results, axis=0)  # (num_samples, num_classes, ...)

    if save_path:
        np.save(save_path, final_attr)

    return final_attr
