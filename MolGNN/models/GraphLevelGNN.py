import pytorch_lightning as pl
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
        self.loss_module = nn.BCEWithLogitsLoss() if self.hparams.c_out == 1 else nn.CrossEntropyLoss()

    def forward(self, data, mode="train"):
        x, edge_index, batch_idx = data.x, data.edge_index, data.batch
        x = self.model(x, edge_index, batch_idx)
        x = x.squeeze(dim=-1)

        if self.hparams.c_out == 1:
            preds = (x > 0).float()
            data.y = data.y.float()
        else:
            preds = x.argmax(dim=-1)
        loss = self.loss_module(x, data.y)
        acc = (preds == data.y).sum().float() / preds.shape[0]
        return loss, acc

    def predict(self, data):
        x, edge_index, batch_idx = data.x, data.edge_index, data.batch
        x = self.model(x, edge_index, batch_idx)
        x = x.squeeze(dim=-1)
        return x

    def configure_optimizers(self, lr: float = 1e-2, weight_decay: float = 0.0):
        # High lr because of small dataset and small model
        optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        loss, acc = self.forward(batch, mode="train")
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss

    def validation_step(self, batch, batch_idx):
        _, acc = self.forward(batch, mode="val")
        self.log('val_acc', acc)

    def test_step(self, batch, batch_idx):
        _, acc = self.forward(batch, mode="test")
        self.log('test_acc', acc)
