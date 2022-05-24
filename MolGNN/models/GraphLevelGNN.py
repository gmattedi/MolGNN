from typing import Tuple

import torch
from sklearn.metrics import matthews_corrcoef, r2_score
from torch import Tensor, nn
from torch_geometric.loader import DataLoader

from MolGNN import models
from MolGNN.utils import logger


class GraphLevelGNN(nn.Module):

    def __init__(self, device: str = 'cpu', **model_kwargs):
        """
        Args:
            device (str)
            **model_kwargs:

        This is a base class that wraps around a GraphNNModel
        that uses GNNModel layers

        GraphGNNModel level args:
            dp_rate_linear (float): Dropout rate before the linear layer (usually much higher than inside the GNN)
            device (str)

        GNNModel level args:
            c_in (int): Dimension of input features
            c_hidden (int): Dimension of hidden features
            c_out (int): Dimension of the output features. Usually number of classes in classification
            num_layers (int): Number of "hidden" graph layers
            layer_name (str): String of the graph layer to use (see gnn_layer_by_name dict)
            dp_rate (float): Dropout rate to apply throughout the network

        Attributes:
            model (nn.Model): Underlying GraphGNN model
            optimizer: Torch optimizer
            criterion: Torch loss module
            metric: Callable that takes the output of the forward pass and the batch labels and
                returns a metric value
            logger: Logger
        """

        super().__init__()

        self.device = device
        self.model = models.GraphGNNModel(device=device, **model_kwargs).to(device)

        self.optimizer = None
        self.criterion = None
        self.metric = None

        self.logger = logger

    def forward(self, batch) -> Tensor:
        """
        Forward pass

        Args:
            batch (DataBatch): batch of PyTorch geometric dataloader

        Returns:
            outputs (Tensor)
        """
        x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch
        outputs = self.model(x, edge_index, batch_idx)
        outputs = outputs.squeeze(dim=-1)
        return outputs

    def step(self, batch) -> Tensor:
        """
        Perform one training step: forward pass, loss evaluation, optimizer step

        Args:
            batch (DataBatch): batch of PyTorch geometric dataloader

        Returns:
            loss (Tensor)
        """
        self.optimizer.zero_grad()

        outputs = self.forward(batch)

        loss = self.criterion(outputs, batch.y)
        loss.backward()
        self.optimizer.step()

        return loss

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, n_epochs: int, log_every_epochs: int) -> Tuple[
        Tensor, float]:
        """
        Train the model

        Args:
            train_loader (DataLoader)
            val_loader (DataLoader)
            n_epochs (int)
            log_every_epochs (int)

        Returns:
            val_loss (Tensor)
            val_metric (float)
        """

        for epoch in range(n_epochs):
            for batch in train_loader:
                _ = self.step(batch)

            if (epoch % log_every_epochs == 0) or (epoch == n_epochs - 1):
                val_loss, val_metric = self.evaluate(val_loader)

                self.logger.info(
                    f"Epoch: {epoch + 1:3d}/{n_epochs:3d} |"
                    f" val loss: {val_loss:8.3f} | val metric: {val_metric:8.3f}"
                )

        val_loss, val_metric = self.evaluate(val_loader)
        return val_loss, val_metric

    def evaluate(self, loader):
        """
        Evaluate the model on a validation/test set

        Args:
            loader (DataLoader)

        Returns:
            val_loss (Tensor)
            val_metric (float)
        """

        train_mode = self.model.training

        if train_mode:
            self.model.eval()

        with torch.no_grad():
            val_outputs, val_y = [], []
            for val_batch in loader:
                val_outputs.append(self.forward(val_batch))
                val_y.append(val_batch.y)

        val_outputs = torch.cat(val_outputs).cpu()
        val_y = torch.cat(val_y).cpu()

        val_loss = self.criterion(val_outputs, val_y)
        val_metric = self.metric(val_y, val_outputs)

        if train_mode:
            self.model.train()

        return val_loss, val_metric


class GraphClassifier(GraphLevelGNN):

    def __init__(self, num_classes, lr: float, weight_decay: float, device: str = 'cpu', **model_kwargs):
        """
        GNN Classifier

        Optimizer: Adam
        Criterion: BCEWithLogsLoss or CrossEntropyLoss

        Args:
            num_classes (int)
            lr (float): Adam lr
            weight_decay (float): Adam weight_decay
            device (str)
            **model_kwargs:
        """

        super().__init__(c_out=num_classes, device=device, **model_kwargs)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.BCEWithLogitsLoss() if num_classes == 1 else nn.CrossEntropyLoss()
        self.metric = matthews_corrcoef

    def evaluate(self, loader, threshold: float = 0.5):
        """
        Evaluate the model on a validation/test set

        Args:
            loader (DataLoader)
            threshold (float): Classification threshold

        Returns:
            val_loss (Tensor)
            val_metric (float)
        """

        train_mode = self.model.training

        if train_mode:
            self.model.eval()

        with torch.no_grad():
            val_outputs, val_y = [], []
            for val_batch in loader:
                val_outputs.append(self.forward(val_batch))
                val_y.append(val_batch.y)

        val_outputs = torch.cat(val_outputs).cpu()
        val_y = torch.cat(val_y).cpu()

        preds = (val_outputs >= threshold).to(int)

        val_loss = self.criterion(val_outputs, val_y)
        val_metric = self.metric(val_y, preds)

        if train_mode:
            self.model.train()

        return val_loss, val_metric


class GraphRegressor(GraphLevelGNN):

    def __init__(self, lr: float, weight_decay: float, device: str = 'cpu', **model_kwargs):
        """
        GNN Regressor

        Optimizer: Adam
        Criterion: MSELoss

        Args:
            num_classes (int)
            lr (float): Adam lr
            weight_decay (float): Adam weight_decay
            device (str)
            **model_kwargs:
        """

        super().__init__(c_out=1, device=device, **model_kwargs)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.MSELoss()
        self.metric = r2_score
