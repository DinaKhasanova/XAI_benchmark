import torch
import numpy as np
from captum.attr import LayerGradCam
from pathlib import Path
from typing import List

def calculate_grad_cam(model, test_loader, conv_layer, device="cuda", save_path=None):

    grad_cam = LayerGradCam(model, conv_layer)
    cam_results = []
    max_seq_len = 0
    temp_results = []

    for batch in test_loader:
        inputs, labels = batch
        inputs = inputs.to(device)

        for i in range(inputs.size(0)):
            input_tensor = inputs[i].unsqueeze(0)
            input_tensor.requires_grad = True

            # Compute Grad-CAM
            attributions = grad_cam.attribute(input_tensor, target=None)
            cam = attributions.squeeze(0).sum(dim=0)
            token_importance = cam.sum(dim=1).detach().cpu().numpy()

            max_seq_len = max(max_seq_len, token_importance.shape[0])
            temp_results.append(token_importance)

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

def calculate_grad_cam_all_layers(model, test_loader, filter_sizes: List[int], save_dir: Path, device="cuba"):
    save_dir.mkdir(parents=True, exist_ok=True)

    model.to(device)

    for idx, size in enumerate(filter_sizes):
        conv_layer = model.convs[idx]
        save_path = save_dir / f"gradcam_filter_{size}.npy"
        print(f"Computing Grad-CAM for filter size {size}")
        cam=calculate_grad_cam(model, test_loader, conv_layer=conv_layer, device=device, save_path=save_path)
        print(f"Saved: {save_path}")
