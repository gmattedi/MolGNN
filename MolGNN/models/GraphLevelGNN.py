import torch
from sklearn.metrics import matthews_corrcoef, r2_score
from torch import nn

from MolGNN import models
from MolGNN.utils import logger


class GraphLevelGNN(nn.Module):

    def __init__(self, **model_kwargs):
        """
        Main GNN model
        Args:
            **model_kwargs:
        """

        super().__init__()

        self.model = models.GraphGNNModel(**model_kwargs)

        self.optimizer = None
        self.criterion = None
        self.metric = None

        self.logger = logger

    def forward(self, data):
        x, edge_index, batch_idx = data.x, data.edge_index, data.batch
        outputs = self.model(x, edge_index, batch_idx)
        outputs = outputs.squeeze(dim=-1)
        return outputs

    def step(self, batch):
        self.optimizer.zero_grad()

        outputs = self.forward(batch)

        loss = self.criterion(outputs, batch.y)
        loss.backward()
        self.optimizer.step()

        return loss

    def fit(self, train_loader, val_loader, n_epochs: int, log_every_epochs: int):

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

    def evaluate(self, val_loader):
        with torch.no_grad():
            val_outputs, val_y = [], []
            for val_batch in val_loader:
                val_outputs.append(self.forward(val_batch))
                val_y.append(val_batch.y)

        val_outputs = torch.cat(val_outputs)
        val_y = torch.cat(val_y)

        val_loss = self.criterion(val_outputs, val_y)
        val_metric = self.metric(val_y, val_outputs)

        return val_loss, val_metric


class GraphClassifier(GraphLevelGNN):

    def __init__(self, num_classes, lr: float, weight_decay: float, **model_kwargs):
        """
        Main GNN model
        Args:
            **model_kwargs:
        """

        super().__init__(c_out=num_classes, **model_kwargs)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.BCEWithLogitsLoss() if num_classes == 1 else nn.CrossEntropyLoss()
        self.metric = matthews_corrcoef

    def evaluate(self, val_loader, threshold: float = 0.5):
        with torch.no_grad():
            val_outputs, val_y = [], []
            for val_batch in val_loader:
                val_outputs.append(self.forward(val_batch))
                val_y.append(val_batch.y)

        val_outputs = torch.cat(val_outputs)
        val_y = torch.cat(val_y)

        preds = (val_outputs >= threshold).to(int)

        val_loss = self.criterion(val_outputs, val_y)
        val_metric = self.metric(val_y, preds)

        return val_loss, val_metric


class GraphRegressor(GraphLevelGNN):

    def __init__(self, lr: float, weight_decay: float, **model_kwargs):
        """
        Main GNN model
        Args:
            **model_kwargs:
        """

        super().__init__(c_out=1, **model_kwargs)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.MSELoss()
        self.metric = r2_score
