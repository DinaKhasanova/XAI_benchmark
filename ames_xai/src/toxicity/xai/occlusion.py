from captum.attr import Occlusion
import torch
import numpy as np

def calculate_occlusion(model, test_loader, num_classes=2, device="cuda", save_path=None):
    device = torch.device(device)
    model = model.to(device)
    occlusion = Occlusion(model)
    occlusion_results = []

    first_batch = next(iter(test_loader))
    input_sample = first_batch[0].to(device)
    input_shape = input_sample.shape  # [B, T, D] or [B, T]

    if len(input_shape) == 2:
        sliding_window_shapes = (1,)
    elif len(input_shape) == 3:
        sliding_window_shapes = (1, input_shape[2])
    else:
        raise ValueError(f"Unsupported input shape for occlusion: {input_shape}")

    print(f"[INFO] Using sliding_window_shapes={sliding_window_shapes}")

    max_sequence_length = max(batch[0].size(1) for batch in test_loader)

    for batch_idx, batch in enumerate(test_loader):
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        labels = labels.long()

        inputs.requires_grad = False

        baseline = torch.zeros_like(inputs)

        attributions = occlusion.attribute(
            inputs,
            baselines=baseline,
            sliding_window_shapes=sliding_window_shapes,
            strides=sliding_window_shapes,
            target=0,
        )

        seq_length = attributions.size(1)
        if seq_length < max_sequence_length:
            padding_needed = max_sequence_length - seq_length
            attributions = torch.cat(
                [attributions, torch.zeros(attributions.size(0), padding_needed, *attributions.shape[2:], device=device)],
                dim=1
            )
            print(f"Padded attributions for batch {batch_idx} to shape: {attributions.shape}")

        occlusion_results.append(attributions.detach().cpu().numpy())

    occlusion_results = np.concatenate(occlusion_results, axis=0)

    if save_path:
        np.save(save_path, occlusion_results)

    return occlusion_results
