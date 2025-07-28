from captum.attr import DeepLift
import torch
import numpy as np

def calculate_deeplift(model, test_loader, num_classes=2, device="cuda", save_path=None):
    device = torch.device(device)
    model = model.to(device)
    deeplift = DeepLift(model)
    dl_results = []

    # Determine max sequence length for padding
    max_sequence_length = max(batch[0].size(1) for batch in test_loader)

    for batch_idx, batch in enumerate(test_loader):
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        inputs.requires_grad = True
        labels = labels.long()

        logits = model(inputs)
        if logits.shape[1] == 1:
            probabilities = torch.sigmoid(logits)

        print(f"logits {logits}")
        print(f"labels {labels}")

        assert labels.min() >= 0 and labels.max() < num_classes, "labels are out of range"

        # Use zero baseline or mean input baseline
        baseline = torch.zeros_like(inputs)

        # Calculate DeepLIFT
        attributions = deeplift.attribute(inputs, baselines=baseline, target=None)

        # Pad to max_sequence_length
        seq_length = attributions.size(1)
        if seq_length < max_sequence_length:
            padding_needed = max_sequence_length - seq_length
            attributions = torch.cat(
                [attributions, torch.zeros(attributions.size(0), padding_needed, *attributions.shape[2:], device=device)],
                dim=1
            )
            print(f"Padded attributions for batch {batch_idx} to shape: {attributions.shape}")

        dl_results.append(attributions.detach().cpu().numpy())

    dl_results = np.concatenate(dl_results, axis=0)

    if save_path:
        np.save(save_path, dl_results)

    return dl_results
