from captum.attr import LayerGradCam
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List

def calculate_grad_cam(model, test_loader, conv_layer, device="cuda", save_path=None):
    device = torch.device(device)
    model = model.to(device)
    gradcam = LayerGradCam(model, conv_layer)
    temp_results = []

    max_seq_len = max(batch[0].size(1) for batch in test_loader)

    for inputs, labels in tqdm(test_loader, desc="Calculating Grad-CAM"):
        inputs = inputs.to(device)
        logits = model(inputs)
        num_classes = logits.shape[1]

        for class_idx in range(num_classes):
            attributions = gradcam.attribute(inputs, target=class_idx)
            # Reduce across the channel dimension if needed
            attributions = attributions.mean(dim=1).squeeze().detach().cpu().numpy()

            if attributions.ndim == 1:
                temp_results.append(attributions)
            else:
                temp_results.extend(attributions)

    cam_results = []
    for cam in temp_results:
        if cam.shape[0] < max_seq_len:
            pad_len = max_seq_len - cam.shape[0]
            cam = np.pad(cam, (0, pad_len), mode="constant", constant_values=0)
        cam_results.append(cam)

    cam_results = np.stack(cam_results)
    if save_path:
        np.save(save_path, cam_results)

    return cam_results

def calculate_grad_cam_all_layers(model, test_loader, filter_sizes: List[int], save_dir: Path, device="cuda"):
    save_dir.mkdir(parents=True, exist_ok=True)
    model.to(device)

    for idx, size in enumerate(filter_sizes):
        conv_layer = model.convs[idx]
        save_path = save_dir / f"gardcam_filter_{size}.npy"
        print(f"Computing Grad-CAM for filter size {size}")
        cam = calculate_grad_cam(model, test_loader, conv_layer=conv_layer, device=device, save_path=save_path)
        print(f"Saved: {save_path}")
