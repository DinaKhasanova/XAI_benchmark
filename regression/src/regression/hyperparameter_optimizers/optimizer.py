import lightning as L
import optuna
import numpy as np
from pathlib import Path
from typing import Optional, Type, TypeVar, Generic
import yaml
import torch
import pandas as pd
import csv
from abc import ABC, abstractmethod
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from regression.config import Config
from regression.data_modules.custom_datamodule import CustomDataModule
from regression.xai.shap import calculate_shap
from regression.xai.ig import calculate_ig
from regression.xai.deeplift import calculate_deeplift
from regression.xai.occlusion import calculate_occlusion
from regression.xai.captum_gradcam_all_layers import calculate_grad_cam_all_layers

T = TypeVar("T", bound=L.LightningModule)

class Optimizer(Generic[T], ABC):
    def __init__(
        self,
        datamodule: CustomDataModule,
        model_class: Type[T],
        result_folder_name: str,
        result_db_name: str,
        training_log_name: str,
        trained_model_name: str,
        optimization_time: int,
        optimized_hyperparameters: Optional[dict] = None,
        percentage_of_validation_set_to_use: float = 0.5,
        max_epochs: int = 20,
    ) -> None:
        self.model_class = model_class
        self.datamodule = datamodule
        self.result_folder_name = result_folder_name
        self.result_db_name = result_db_name
        self.training_log_name = training_log_name
        self.trained_model_name = trained_model_name
        self.optimization_time = optimization_time
        self.optimized_hyperparameters = optimized_hyperparameters
        self.percentage_of_validation_set_to_use = percentage_of_validation_set_to_use
        self.max_epochs = max_epochs
        directory = Config.get_directory()
        self.subfolder_path = directory / self.result_folder_name
        self.subfolder_path.mkdir(exist_ok=True)

    @abstractmethod
    def objective(self, trial: optuna.trial.Trial) -> float:
        pass

    @abstractmethod
    def get_optimized_hyperparameters(self) -> None:
        pass

    @abstractmethod
    def train_optimized_model(self) -> None:
        pass

    def save_best_hyperparameter(self) -> None:
        file_path = self.subfolder_path / "optimized_hyperparameters.yaml"
        with open(file_path, "w") as yaml_file:
            yaml.dump(self.optimized_hyperparameters, yaml_file, default_flow_style=False)

    def test_model(self) -> None:
        trained_model_path = self.subfolder_path / "model.ckpt"
        model = self.model_class.load_from_checkpoint(trained_model_path)
        model.eval()
        self.datamodule.setup(stage="test")
        test_dataloader = self.datamodule.test_dataloader()

        y_pred = []
        y_true = []
        with torch.no_grad():
            for x, y in test_dataloader:
                if isinstance(x, torch.Tensor):
                    predictions = model(x.to(model.device))
                else:
                    predictions = model(x)
                if predictions.dim() == 0:
                    predictions = predictions.unsqueeze(0)
                assert predictions.shape == y.shape, "Predictions and labels dont have the same shape."
                for pred, label in zip(predictions, y):
                    y_pred.append(pred.detach().cpu())
                    y_true.append(label.detach().cpu())

        y_pred_numpy = torch.stack(y_pred).numpy()
        y_true_numpy = torch.stack(y_true).numpy()

        mae = mean_absolute_error(y_true_numpy, y_pred_numpy)
        mse = mean_squared_error(y_true_numpy, y_pred_numpy)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true_numpy, y_pred_numpy)

        metrics_path = self.subfolder_path / "final_test_metrics.csv"
        with open(metrics_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["mae", "rmse", "r2"])
            writer.writerow([mae, rmse, r2])

        functional_groups = ['halide', 'amine', 'carboxylic_esters', 'ketone', 'ether',
                             'carboxylic_acid', 'arene', 'carboxylic_amides', 'nitro', 'hydroxyl']

        df_true = pd.DataFrame(y_true_numpy, columns=[f"true_{fg}" for fg in functional_groups])
        df_pred = pd.DataFrame(y_pred_numpy, columns=[f"pred_{fg}" for fg in functional_groups])
        df_all = pd.concat([df_true, df_pred], axis=1)

        per_mol_path = self.subfolder_path / "predictions_per_molecule.csv"
        df_all.to_csv(per_mol_path, index=False)

    def explain_model(self) -> None:
        trained_model_path = self.subfolder_path / "model.ckpt"
        model = self.model_class.load_from_checkpoint(trained_model_path)
        model.eval()
        self.datamodule.setup(stage="test")
        test_dataloader = self.datamodule.test_dataloader()

        ig_path = self.subfolder_path / "ig.npy"
        shap_path = self.subfolder_path / "shap.npy"
        deeplift_path = self.subfolder_path / "deeplift.npy"
        occlusion_path = self.subfolder_path / "occlusion.npy"
        gradcam_dir = self.subfolder_path / "gradcams"
        filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]

        print("Calculating XAI methods...")
        calculate_shap(model, test_dataloader, save_path=shap_path)
        calculate_ig(model, test_dataloader, save_path=ig_path)
        calculate_grad_cam_all_layers(model.model, test_loader=test_dataloader,
                                      filter_sizes=filter_sizes, save_dir=gradcam_dir, device="cuda")
        calculate_deeplift(model, test_dataloader, save_path=deeplift_path)
        calculate_occlusion(model, test_dataloader, save_path=occlusion_path)

        print("Saved XAI outputs:")
        print(ig_path)
        print(gradcam_dir)
        print(shap_path)
        print(deeplift_path)
        print(occlusion_path)
