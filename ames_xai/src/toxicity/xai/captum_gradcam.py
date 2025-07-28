import torch
import numpy as np
from captum.attr import LayerGradCam

def calculate_grad_cam(model, test_loader, conv_layer, device="cuda", save_path=None):
    model.to(device)
    model.eval()

    grad_cam = LayerGradCam(model, conv_layer)
    cam_results = []
    max_seq_len = 0
    temp_results = []

    for batch in test_loader:
        inputs, labels = batch
        print(f"Batch size: {inputs.size(0)} molecules")
        print(f"Input shape: {inputs.shape}")
        inputs = inputs.to(device)
        for i in range(inputs.size(0)):
            input_tensor = inputs[i].unsqueeze(0)  # [1, 1, seq_len, emb_dim]
            input_tensor.requires_grad = True

            # Compute Grad-CAM
            attributions = grad_cam.attribute(input_tensor, target=None)  # shape: [1, n_channels, H, W]
            cam = attributions.squeeze(0).sum(dim=0)  # shape: [H, W]
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
