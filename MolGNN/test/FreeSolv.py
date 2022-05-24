import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader

from MolGNN import MolToGraph, models

"""
Test MolGNN on FreeSolv, with random split
"""

# ---- Config ------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# ------------------------

solv = pd.read_csv('FreeSolv.tsv', sep=';')
smiles = solv.SMILES.values

train_idx, val_idx = train_test_split(range(len(smiles)), test_size=0.2, random_state=42)
smiles_train, smiles_val = smiles[train_idx], smiles[val_idx]

print('----------------- CLASSIFICATION -----------------')
y = (
        solv['experimental value (kcal/mol)'].values <= solv['experimental value (kcal/mol)'].median()
).astype(int)
y_train, y_val = y[train_idx], y[val_idx]

train_loader = DataLoader(
    dataset=MolToGraph.create_pyg_data_lst(smiles_train, y_train),
    batch_size=32
)
val_loader = DataLoader(
    dataset=MolToGraph.create_pyg_data_lst(smiles_val, y_val),
    batch_size=32
)

model = models.GraphClassifier(
    num_classes=1,
    c_in=79,
    c_hidden=256,
    num_layers=3,
    dp_rate_linear=0.5,
    dp_rate=0.0,
    lr=1e-2, weight_decay=0
)

model.fit(train_loader, val_loader, n_epochs=100, log_every_epochs=10)

print('----------------- REGRESSION ---------------------')

y = solv['experimental value (kcal/mol)'].values
y_train, y_val = y[train_idx], y[val_idx]

train_loader = DataLoader(
    dataset=MolToGraph.create_pyg_data_lst(smiles_train, y_train),
    batch_size=32
)
val_loader = DataLoader(
    dataset=MolToGraph.create_pyg_data_lst(smiles_val, y_val),
    batch_size=32
)

model = models.GraphRegressor(
    c_in=79,
    c_hidden=256,
    num_layers=3,
    dp_rate_linear=0.5,
    dp_rate=0.0,
    lr=1e-2, weight_decay=0
)

model.fit(train_loader, val_loader, n_epochs=100, log_every_epochs=10)
