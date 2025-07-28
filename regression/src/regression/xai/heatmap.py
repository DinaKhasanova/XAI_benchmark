import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def generate_heatmap(csv_path, output_path="/home/dina/xai_methods/src/xai_benchmark/model/heatmap.png"):
    df = pd.read_csv(csv_path)
    
    # Extract numeric values for heatmap
    numeric_df = df.set_index("method_pair").applymap(
        lambda x: float(str(x).split(" ")[0]) if isinstance(x, str) and "±" in x else pd.to_numeric(x, errors='coerce')
    )

    plt.figure(figsize=(12, 8))
    sns.heatmap(numeric_df, annot=True, fmt=".4f", cmap="viridis")
    plt.title("Cosine Distance Comparison Heatmap")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Heatmap saved to {output_path}")

if __name__ == "__main__":
    csv_path = "/home/dina/xai_methods/src/xai_benchmark/model/cosine_distance_summary.csv"
    generate_heatmap(csv_path)
