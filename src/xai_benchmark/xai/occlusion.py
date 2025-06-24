from captum.attr import Occlusion
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

def calculate_occlusion(model, test_loader, device="cuda", save_path=None):
    device = torch.device(device)
    model = model.to(device)
    occlusion = Occlusion(model)
    occ_results = []

    max_sequence_length = max(batch[0].size(1) for batch in test_loader)

    for batch_idx, (inputs, labels) in enumerate(tqdm(test_loader, desc="Calculating Occlusion")):
        inputs, labels = inputs.to(device), labels.to(device)
        inputs.requires_grad = True

        logits = model(inputs)
        num_classes = logits.shape[1]

        sliding_window_shapes = (1, inputs.size(2))
        baselines = torch.zeros_like(inputs)

        batch_result = []
        for class_idx in range(num_classes):
            attributions = occlusion.attribute(
                inputs,
                baselines=baselines,
                target=class_idx,
                sliding_window_shapes=sliding_window_shapes
            )

            seq_length = attributions.size(1)
            if seq_length < max_sequence_length:
                padding_needed = max_sequence_length - seq_length
                attributions = torch.cat(
                    [attributions, torch.zeros(attributions.size(0), padding_needed, *attributions.shape[2:], device=device)],
                    dim=1
                )

            batch_result.append(attributions.detach().cpu().numpy())

        batch_result = np.stack(batch_result, axis=1)  # shape: (batch_size, num_classes, seq_len, embed_dim)
        occ_results.append(batch_result)

    occ_results = np.concatenate(occ_results, axis=0)

    if save_path:
        np.save(save_path, occ_results)

    return occ_results
