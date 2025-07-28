import numpy as np
from lime.lime_tabular import LimeTabularExplainer
import torch
import torch.nn.functional as F

def calculate_lime(model, test_loader, num_classes=2, num_samples=100, save_path=None, device="cuda"):
    device = torch.device(device)
    model = model.to(device)
    model.eval()

    all_embeddings = []
    all_labels = []
    max_length = 0

    for embeddings, labels in test_loader:
        embeddings = embeddings.to(device)
        all_embeddings.append(embeddings)
        all_labels.append(labels)
        max_length = max(max_length, embeddings.size(1))

    # Pad sequences to max_length
    padded_embeddings = [
        F.pad(emb, (0, 0, 0, max_length - emb.size(1)))
        for emb in all_embeddings
    ]
    all_embeddings = torch.cat(padded_embeddings, dim=0)  #concatenate along the batch axis
    all_labels = torch.cat(all_labels, dim=0)
    
    N, L, D = all_embeddings.shape
    embeddings_np = all_embeddings.cpu().numpy()
    embeddings_flat = embeddings_np.reshape((N, L * D))

    # Initialize LIME with background data
    background_data = embeddings_flat[:num_samples]
    explainer = LimeTabularExplainer(
        background_data,
        mode='classification',
        feature_names=[f"f{i}" for i in range(L * D)],
        discretize_continuous=False
    )

    lime_attributions = []

    def predict_fn(x_flat):
        x_tensor = torch.tensor(x_flat, dtype=torch.float32).to(device)
        x_tensor = x_tensor.view(-1, L, D)
        with torch.no_grad():
            logits = model(x_tensor)
            if logits.shape[1] == 1:
                probs = torch.sigmoid(logits)
                probs = torch.cat([1 - probs, probs], dim=1)
            else:
                probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()

    for i in range(N):
        instance = embeddings_flat[i]

        # Predict class to compare with SHAP/IG
        input_tensor = torch.tensor(instance, dtype=torch.float32).to(device).view(1, L, D)
        with torch.no_grad():
            logits = model(input_tensor)
            predicted_class = torch.argmax(logits, dim=1).item()

        # Explain instance
        exp = explainer.explain_instance(instance, predict_fn, num_features=L * D, top_labels=num_classes)

        # Get attribution only for the predicted class
        instance_attribution = np.zeros((L * D))
        for idx, val in exp.as_map()[predicted_class]:
            instance_attribution[idx] = val

        # Reshape to [L, D] and store
        instance_attribution = instance_attribution.reshape(L, D)
        lime_attributions.append(instance_attribution)

    if save_path:
        np.save(save_path, lime_attributions)
        print(f"LIME attributions saved to {save_path}")

    return lime_attributions
