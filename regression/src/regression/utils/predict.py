import lightning as L
from pathlib import Path
from typing import Tuple, Optional, Dict
import numpy as np
from numpy.typing import NDArray
from regression.data_modules.datasets_and_collate_functions import (
    GraphDataset,
    mol_collate_fn,
    CNNDataset,
    CNN_collate_fn,
)
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
import rdkit.Chem as Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.rdMolDescriptors import GetMACCSKeysFingerprint
from regression.models.MLP import MLPLightning
from regression.models.MPNN import MPNNLightning
from regression.utils.convert_representations import convert_dataset


class TestPredictor:

    def __init__(
        self, representation: str, model: MLPLightning | MPNNLightning, datafile: Path
    ) -> None:
        self.representation = representation
        self.model = model
        self.model.eval()
        self.datafile = datafile
        self.input_vec_dim: Optional[int] = None

    def load_data(self) -> pd.DataFrame:
        data = pd.read_csv(self.datafile)
        assert "smiles" in data.columns, "'smiles' column is missing"
        return data

    def get_y_pred_and_y_true(self) -> Dict[str, NDArray[np.float64]]:
        data = self.load_data()
        dataset = self.convert_dataset(data=data)
        if self.representation == "graph":
            collate_function = mol_collate_fn
        elif self.representation == "transformer_matrix":
            collate_function = CNN_collate_fn
        else:
            collate_function = None
            assert (
                self.input_vec_dim == self.model.input_vec_dim
            ), "Input vector dimension of model does not line up with the model representation."
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=32,
            shuffle=False,
            num_workers=8,
            prefetch_factor=4,
            collate_fn=collate_function,
        )
        y_true = []
        y_pred = []
        with torch.no_grad():
            for x, y in iter(dataloader):
                if isinstance(x, torch.Tensor):
                    preds = self.model(x.to(self.model.device))
                else:
                    preds = self.model(x)
                assert (
                    preds.shape == y.shape
                ), "Predictions and labels dont have the same shape."
                for pred, label in zip(preds, y):
                    y_pred.append(pred.detach().cpu())
                    y_true.append(label.detach().cpu())
        return {"y_pred": torch.stack(y_pred).numpy(), "y_true": torch.stack(y_true).numpy()}

    def convert_dataset(
        self, data: pd.DataFrame
    ) -> TensorDataset | GraphDataset | CNNDataset:
        dataset, input_vec_dim = convert_dataset(
            data=data, representation=self.representation
        )
        self.input_vec_dim = input_vec_dim
        return dataset


class BaselinePredictor:

    def __init__(self, train_datafile: Path, test_datafile: Path) -> None:
        self.train_datafile = train_datafile
        self.test_datafile = test_datafile

    def load_data(self, dataset: str) -> pd.DataFrame:
        if dataset == "train":
            data = pd.read_csv(self.train_datafile)
        elif dataset == "test":
            data = pd.read_csv(self.test_datafile)
        else:
            raise NotImplementedError
        assert "smiles" in data.columns, "'smiles' column is missing"
        return data

    def get_class_imbalance(self) -> float:
        data = self.load_data("train")
        label_columns = data.columns.drop("smiles")
        class_imbalance = data[label_columns].mean().mean()
        return class_imbalance

    def get_y_pred_and_y_true(self):
        test_data = self.load_data("test")
        label_columns = test_data.columns.drop("smiles")
        y_true = test_data[label_columns].values

        # Generate baseline predictions as mean of training data for regression
        train_data = self.load_data("train")
        mean_values = train_data[label_columns].mean().values
        y_pred = np.tile(mean_values, (len(test_data), 1))

        return {"y_pred": np.array(y_pred), "y_true": np.array(y_true)}
