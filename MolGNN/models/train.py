import os
import warnings
from typing import Tuple, Dict

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from MolGNN import models

warnings.filterwarnings(action='ignore')


def train_graph_classifier(
        train_loader,
        val_loader,
        num_node_features: int,
        num_classes: int,
        model_name: str,
        device: str,
        checkpoint_path: str,
        pl_random_seed: int = 42,
        **model_kwargs) -> Tuple[pl.LightningModule, Dict[str, float]]:
    pl.seed_everything(pl_random_seed)

    # Create a PyTorch Lightning trainer with the generation callback
    root_dir = os.path.join(checkpoint_path, "GraphLevel" + model_name)
    os.makedirs(root_dir, exist_ok=True)
    trainer = pl.Trainer(
        default_root_dir=root_dir,
        callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc")],
        gpus=1 if str(device).startswith("cuda") else 0,
        max_epochs=500,
        progress_bar_refresh_rate=0)
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    c_out = 1 if num_classes == 2 else num_classes
    model = models.GraphClassifier(c_in=num_node_features,
                                   c_out=c_out,
                                   **model_kwargs)
    trainer.fit(model, train_loader, val_loader)
    model = models.GraphClassifier.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    # Test best model on validation and test set
    train_result = trainer.test(model, train_loader, verbose=False)
    test_result = trainer.test(model, val_loader, verbose=False)
    result = {"test": test_result[0]['test_acc'], "train": train_result[0]['test_acc']}

    return model, result

# TODO r2 return NaN
def train_graph_regressor(
        train_loader,
        val_loader,
        num_node_features: int,
        model_name: str,
        device: str,
        checkpoint_path: str,
        pl_random_seed: int = 42,
        **model_kwargs) -> Tuple[pl.LightningModule, Dict[str, float]]:
    pl.seed_everything(pl_random_seed)

    # Create a PyTorch Lightning trainer with the generation callback
    root_dir = os.path.join(checkpoint_path, "GraphLevel" + model_name)
    os.makedirs(root_dir, exist_ok=True)
    trainer = pl.Trainer(
        default_root_dir=root_dir,
        callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_r2")],
        gpus=1 if str(device).startswith("cuda") else 0,
        max_epochs=500,
        progress_bar_refresh_rate=0)
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    model = models.GraphRegressor(c_in=num_node_features,
                                  c_out=1,
                                  **model_kwargs)
    trainer.fit(model, train_loader, val_loader)
    model = models.GraphRegressor.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    # Test best model on validation and test set
    train_result = trainer.test(model, train_loader, verbose=False)
    test_result = trainer.test(model, val_loader, verbose=False)
    result = {"test": test_result[0]['test_r2'], "train": train_result[0]['test_r2']}

    return model, result
