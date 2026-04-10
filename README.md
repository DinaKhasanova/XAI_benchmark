# Benchmarking explainable AI methods for toxicophore detection and toxicity prediction

A benchmarking workflow for analyzing the robustness, consistency, and agreement of explainability techniques applied to molecular property prediction models.

## Getting Started

This section will help you set up the required environment and run the project locally.

### Prerequisites

You need to have an environment with python version 3.12. This can be created using conda.

```
$ conda env create -f environment.yml
$ conda activate xai_env 
```

### Installation


Clone the repository:

```
$ git clone https://github.com/DinaKhasanova/XAI_benchmark.git
$ cd XAI_benchmark
```

## Usage

The project can be run from the terminal by executing the main Python script:

```
$ python main.py install-model transformer_matrix #train the transformer
$ python main.py #train and test CNN
```
#### Optional Arguments
`--dataset`: Specifies the dataset to use. Choices are:
- `ames` (default)
- `file` (use your own dataset; requires `--dataset_path`)

`--dataset_path`: Path(s) to your custom dataset file(s). Required if `--dataset file` is selected. The dataset must have a 'smiles' column and a binary 'label' column.

`--split`: Split ratios for training, validation, and testing datasets. Provide three float values that sum to 1.0. Default: `0.7 0.1 0.2`

`--hyperparameter_optimization_time`: Time in seconds allocated for hyperparameter optimization during model training.
Default: `600`

## Test Output

When you run the test, the results are saved in the `result_folder` you specified. The folder structure contains the following files and directories:

#### `model/`
- **model_performance.pdf**: ROC curve of the model performance in the binary classification task.
- **model.ckpt**: The checkpoint file for the final model that was trained with the optimized hyperparameters.
- **optimization.db**: Database containing the optimization history during hyperparameter tuning.
- **optimized_hyperparameters.yaml**: A YAML file that contains the optimized hyperparameters after the search process.
- **deeplift.npy**: Attribution scores generated using DeepLIFT.
- **ig.npy**: Integrated Gradients attribution scores.
- **occlusion.npy**: Occlusion sensitivity attributions.
- **shap.npy**: SHAP-based feature attributions.
- **gradcam_combined.npy**: Attribution map combining layer-level Grad-CAM maps.
- **gradcams**: layer-wise Grad-CAM visualizations from the CNN.
- **xai_distances_summary_all.csv**: Pairwise similarity metrics comparing attribution methods across molecules.


#### `data_dir/`
- **train.csv**, **validation.csv**, **test.csv**: These files contain the datasets used for training, validation, and testing. Will vary when re-running the package since the split is random.

#### `model_config.yaml`
- A configuration file that summarizes all model related configurations.



## Citation

This repository is part of the paper "Benchmarking explainable AI methods for toxicophore detection and toxicity prediction". \\
Published paper: TODO \\
Bibtex: TODO



