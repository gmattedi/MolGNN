import pytorch_lightning as pl
from sklearn.metrics import r2_score
from torch import nn, optim

from MolGNN import models


class GraphLevelGNN(pl.LightningModule):

    def __init__(self, **model_kwargs):
        """
        Main GNN model
        Args:
            **model_kwargs:
        """

        super().__init__()
        # Saving hyperparameters
        self.save_hyperparameters()

        self.model = models.GraphGNNModel(**model_kwargs)

        self.loss_module = None

    def forward(self, data):
        x, edge_index, batch_idx = data.x, data.edge_index, data.batch
        outputs = self.model(x, edge_index, batch_idx)
        outputs = outputs.squeeze(dim=-1)
        return outputs

    def metric_fn(self, preds, batch):
        raise NotImplementedError

    def configure_optimizers(self, lr: float = 1e-2, weight_decay: float = 0.0):
        # High lr because of small dataset and small model
        optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def test_step(self, batch, batch_idx):
        raise NotImplementedError


class GraphClassifier(GraphLevelGNN):

    def __init__(self, **model_kwargs):
        """
        Main GNN model
        Args:
            **model_kwargs:
        """

        super().__init__(**model_kwargs)
        # Saving hyperparameters
        self.save_hyperparameters()

        self.model = models.GraphGNNModel(**model_kwargs)

        self.loss_module = nn.BCEWithLogitsLoss() if self.hparams.c_out == 1 else nn.CrossEntropyLoss()

    def metric_fn(self, y_true, preds):
        acc = (preds == y_true).sum().float() / preds.shape[0]
        return acc

    def _step(self, batch, batch_idx):
        outputs = self.forward(batch)
        loss = self.loss_module(outputs, batch.y)

        if self.hparams.c_out == 1:
            preds = (outputs > 0).float()
            batch.y = batch.y.float()
        else:
            preds = outputs.argmax(dim=-1)
        acc = self.metric_fn(batch.y, preds)

        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self._step(batch, batch_idx)

        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._step(batch, batch_idx)

        self.log('val_loss', loss)
        self.log('val_acc', acc)
        return loss

    def test_step(self, batch, batch_idx):
        loss, acc = self._step(batch, batch_idx)

        self.log('test_loss', loss)
        self.log('test_acc', acc)
        return loss


class GraphRegressor(GraphLevelGNN):

    def __init__(self, **model_kwargs):
        """
        Main GNN model
        Args:
            **model_kwargs:
        """

        super().__init__(**model_kwargs)
        # Saving hyperparameters
        self.save_hyperparameters()

        self.model = models.GraphGNNModel(**model_kwargs)

        self.loss_module = nn.MSELoss()

    def metric_fn(self, y_true, preds):
        r2 = r2_score(y_true, preds)
        return r2

    def _step(self, batch, batch_idx):
        outputs = self.forward(batch)
        loss = self.loss_module(outputs, batch.y)

        r2 = self.metric_fn(
            batch.y.detach().cpu().numpy(),
            outputs.detach().cpu().numpy()
        )

        return loss, r2

    def training_step(self, batch, batch_idx):
        loss, r2 = self._step(batch, batch_idx)

        self.log('train_loss', loss)
        self.log('train_r2', r2)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, r2 = self._step(batch, batch_idx)

        self.log('val_loss', loss)
        self.log('val_r2', r2)
        return loss

    def test_step(self, batch, batch_idx):
        loss, r2 = self._step(batch, batch_idx)

        self.log('test_loss', loss)
        self.log('test_r2', r2)
        return loss
