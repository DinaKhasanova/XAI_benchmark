from xai_benchmark.data_modules.custom_datamodule import (
    CustomDataModule,
)
from xai_benchmark.config import Config
import pandas as pd
from typing import List, Optional
import torch
from rdkit import Chem
from xai_benchmark.utils.csv_file_operations import save_csv
from pathlib import Path


class UserDataModule(CustomDataModule):
    """Class for data management of Ames dataset. Desired representation of molecules must be specified in the initialization"""

    def __init__(
        self,
        representation: str,
        dataset_paths: List[str],
        split: List[float],
        batch_size: int = 64,
        num_workers: int = 8,
        prefetch_factor: int = 8,
    ):
        super().__init__(
            representation=representation,
            dataset_name="user_provided_dataset",
            split=split,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
        )
        self.dataset_paths = dataset_paths
        self.train_data: Optional[pd.DataFrame] = None
        self.validation_data: Optional[pd.DataFrame] = None
        self.test_data: Optional[pd.DataFrame] = None

    def prepare_data(self) -> None:
        train_file = self.data_dir / "train.csv"
        validation_file = self.data_dir / "validation.csv"
        test_file = self.data_dir / "test.csv"
        if not (
            train_file.exists() and validation_file.exists() and test_file.exists()
        ):
            if len(self.dataset_paths) == 1:
                print(
                    "Found one dataset. This will be split into train val and test according to the split"
                )
                data = pd.read_csv(self.dataset_paths[0])
                self.sanity_check(data)
                data = data.sample(frac=1).reset_index(drop=True)
                self.train_data = data.iloc[: int(len(data) * self.split[0])]
                self.validation_data = data.iloc[
                    int(len(data) * self.split[0]) : int(
                        len(data) * (self.split[0] + self.split[1])
                    )
                ]
                self.test_data = data.iloc[
                    int(len(data) * (self.split[0] + self.split[1])) :
                ]
            # elif len(self.dataset_paths) == 2:
            #     print(f"Found 2 datasets. Using the first one ({self.dataset_paths[0]}) as training data and the second one {self.dataset_paths[1]}) as validation data.")
            #     train_data = pd.read_csv(self.dataset_paths[0])
            #     self.sanity_check(train_data)
            #     self.train_data = train_data
            #     validation_data = pd.read_csv(self.dataset_paths[1])
            #     self.sanity_check(validation_data)
            #     self.validation_data = validation_data
            elif len(self.dataset_paths) == 3:
                print(
                    f"Found 3 datasets. Using the first one ({self.dataset_paths[0]}) as training data, the second one ({self.dataset_paths[1]}) as validation data, and the last one ({self.dataset_paths[2]}) as test data."
                )
                train_data = pd.read_csv(self.dataset_paths[0])
                self.sanity_check(train_data)
                self.train_data = train_data
                validation_data = pd.read_csv(self.dataset_paths[1])
                self.sanity_check(validation_data)
                self.validation_data = validation_data
                test_data = pd.read_csv(self.dataset_paths[2])
                self.sanity_check(test_data)
                self.test_data = test_data
            else:
                raise ValueError(
                    f"Provided {len(self.dataset_paths)} dataset paths. Expected 1 or 3."
                )
            save_csv(self.train_data, "train.csv")
            save_csv(self.validation_data, "validation.csv")
            save_csv(self.test_data, "test.csv")

    def setup(self, stage: str) -> None:
        if stage == "fit":
            train_data = pd.read_csv(self.data_dir / "train.csv")
            label_columns = train_data.columns.drop("smiles")

            positives = train_data[label_columns].sum(axis=0)
            negatives = len(train_data) - positives

            Config.set_pos_weights(
                torch.tensor([negatives / positives], dtype=torch.float32)
            )
            self.train_dataset = self.convert_dataset(train_data)
            validation_data = pd.read_csv(self.data_dir / "validation.csv")
            self.validation_dataset = self.convert_dataset(validation_data)
        if stage == "test":
            test_data = pd.read_csv(self.data_dir / "test.csv")
            self.test_dataset = self.convert_dataset(test_data)

    def sanity_check(self, data: pd.DataFrame) -> None:
        if "smiles" not in data.columns:
            raise ValueError("Missing 'smiles' column in provided data.")

        label_columns = data.columns.drop("smiles")

        if not all(data[label_columns].isin([0, 1]).all()):
            raise ValueError(
                f"Label columns must contain only 0s and 1s. Found other values in one or more functional group columns."
            )

        if len(data) < 10:
            raise ValueError(
                f"Found less than 10 molecules in the provided data. That's way too few. Amount is {len(data)}."
            )

        invalid_smiles = data["smiles"].apply(lambda x: Chem.MolFromSmiles(x) is None)
        if invalid_smiles.any():
            invalid_entries = data["smiles"][invalid_smiles]
            raise ValueError(
                f"The 'smiles' column contains these invalid SMILES strings: {invalid_entries.tolist()}"
            )