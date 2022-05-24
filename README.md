# GNN-based QSAR models

### An example of Graph Neural Networks for QSAR modelling

The base models are adapted
from [this tutorial](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial7/GNN_overview.html),
part of the [UvA Deep Learning](https://uvadlc.github.io/) course.

The featurisation of molecules as `torch_geometric` objects is taken
from [this blog post](https://www.blopig.com/blog/2022/02/how-to-turn-a-smiles-string-into-a-molecular-graph-for-pytorch-geometric/)
by the Oxford Protein Informatics Group.

# Walkthrough

Here is an example where the models are used to predict experimental hydration free energy of
the [FreeSolv](https://github.com/MobleyLab/FreeSolv) dataset.

## Import the necessary modules

```python
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

from MolGNN import MolToGraph
from MolGNN.models import GraphClassifier, GraphRegressor

device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

## Load the data
```python
solv = pd.read_csv('MolGNN/test/FreeSolv.tsv', sep=';')
smiles = solv.SMILES.values

train_idx, val_idx = train_test_split(range(len(smiles)), test_size=0.2, random_state=42)
smiles_train, smiles_val = smiles[train_idx], smiles[val_idx]
```

## Train a GNN classifier
We label compounds based on whether the experimental hydration energy is above the median,
so to have an artifically perfectly balanced dataset.
The performance metric used here is MCC.

```python
y = (
        solv['experimental value (kcal/mol)'].values >= solv['experimental value (kcal/mol)'].median()
).astype(int)
y_train, y_val = y[train_idx], y[val_idx]

# Featurize the molecules and build dataloaders
train_loader = DataLoader(
    dataset=MolToGraph.create_pyg_data_lst(smiles_train, y_train, device=device),
    batch_size=32
)
val_loader = DataLoader(
    dataset=MolToGraph.create_pyg_data_lst(smiles_val, y_val, device=device),
    batch_size=32
)

# Initialize the model
model = GraphClassifier(
    num_classes=1,
    c_in=79,
    c_hidden=256,
    num_layers=3,
    dp_rate_linear=0.5,
    dp_rate=0.0,
    lr=1e-2, weight_decay=0,
    device=device
)

# Train the model
val_loss, val_metric = model.fit(train_loader, val_loader, n_epochs=100, log_every_epochs=10)
```
```2022-05-24 17:08:47 INFO     Epoch: 100/100 | val loss:    0.374 | val metric:    0.837```

## Train a GNN regressor
The performance metric is R2.

```python
y = solv['experimental value (kcal/mol)'].values
y_train, y_val = y[train_idx], y[val_idx]

# Featurize the molecules and build dataloaders
train_loader = DataLoader(
    dataset=MolToGraph.create_pyg_data_lst(smiles_train, y_train, device=device),
    batch_size=32
)
val_loader = DataLoader(
    dataset=MolToGraph.create_pyg_data_lst(smiles_val, y_val, device=device),
    batch_size=32
)

# Initialize the model
model = GraphRegressor(
    c_in=79,
    c_hidden=256,
    num_layers=3,
    dp_rate_linear=0.5,
    dp_rate=0.0,
    lr=1e-2, weight_decay=0,
    device=device
)

# Train the model
val_loss, val_metric = model.fit(train_loader, val_loader, n_epochs=100, log_every_epochs=10)
```
```2022-05-24 17:06:36 INFO     Epoch: 100/100 | val loss:    1.672 | val metric:    0.894```