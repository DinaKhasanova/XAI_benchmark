from captum.attr import DeepLift
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

def calculate_deeplift(model, test_loader, device="cuda", save_path=None):
    device = torch.device(device)
    model = model.to(device)
    deeplift = DeepLift(model)
    all_dl_values = []

    max_sequence_length = max(batch[0].size(1) for batch in test_loader)

    for batch_idx, (inputs, labels) in enumerate(tqdm(test_loader, desc="Calculating DeepLIFT")):
        inputs = inputs.to(device)
        inputs.requires_grad = True

        logits = model(inputs)
        num_classes = logits.shape[1]
        batch_size = inputs.size(0)
        embedding_dim = inputs.size(2)

        baseline = torch.zeros_like(inputs)
        class_level_attributions = []

        for class_idx in range(num_classes):
            attributions = deeplift.attribute(inputs, baselines=baseline, target=class_idx)
            if attributions.size(1) < max_sequence_length:
                padding_needed = max_sequence_length - attributions.size(1)
                pad_tensor = torch.zeros((batch_size, padding_needed, embedding_dim), device=device)
                attributions = torch.cat([attributions, pad_tensor], dim=1)
            class_level_attributions.append(attributions.detach().cpu().numpy())

        class_level_attributions = np.stack(class_level_attributions, axis=0)
        class_level_attributions = np.transpose(class_level_attributions, (1, 0, 2, 3))
        all_dl_values.append(class_level_attributions)

    final_result = np.concatenate(all_dl_values, axis=0)

    if save_path:
        np.save(save_path, final_result)

    return final_result
