import torch
import shap
import torch.nn.functional as F
import numpy as np

def calculate_shap_deep(model, test_loader, num_samples=100, save_path=None, device="cuda"):
    device = torch.device(device)
    model = model.to(device)
    model.eval()

    all_embeddings = []
    all_labels = []
    max_length = 0

    for batch in test_loader:
        if isinstance(batch, tuple):
            embeddings, labels = batch
            all_labels.append(labels)
        else:
            embeddings = batch
        embeddings = embeddings.to(device)
        all_embeddings.append(embeddings)
        max_length = max(max_length, embeddings.size(1))

    # Pad all sequences to the max length
    padded_embeddings = [
        F.pad(emb, (0, 0, 0, max_length - emb.size(1)))
        for emb in all_embeddings
    ]
    all_embeddings = torch.cat(padded_embeddings, dim=0)  # [N, L, D]
    if all_labels:
        all_labels = torch.cat(all_labels, dim=0).to(device)

    background_data = all_embeddings[:num_samples]
    background_data = background_data.to(device)

    print(f"Using SHAP DeepExplainer")
    print(f"Model is on device: {next(model.parameters()).device}")
    print(f"Input embeddings shape: {all_embeddings.shape}")
    print(f"Background shape: {background_data.shape}")

    # Create SHAP DeepExplainer
    explainer = shap.DeepExplainer(model, background_data)

    try:
        shap_values = explainer.shap_values(all_embeddings)  # list of [class_id][N, L, D]
        print(f"SHAP (DeepExplainer) values calculated. Shapes: {[v.shape for v in shap_values]}")
    except Exception as e:
        print(f"Error during SHAP DeepExplainer calculation: {e}")
        raise

    # Select attributions for predicted class only
    with torch.no_grad():
        logits = model(all_embeddings)
        predicted_classes = torch.argmax(logits, dim=1).cpu().numpy()

    selected_shap_values = np.zeros_like(shap_values[0])
    for i, class_id in enumerate(predicted_classes):
        selected_shap_values[i] = shap_values[class_id][i]

    if save_path:
        np.save(save_path, selected_shap_values)
        print(f"SHAP DeepExplainer values saved to {save_path}")

    return selected_shap_values  # shape: [N, L, D]
