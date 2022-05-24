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

print('----------------- CLASSIFICATION -----------------')
CHECKPOINT_PATH = 'train_cls/'
# Load the data and process in into PyG Data objects
solv = pd.read_csv('FreeSolv.tsv', sep=';')
smiles = solv.SMILES.values
y = (
        solv['experimental value (kcal/mol)'].values <= solv['experimental value (kcal/mol)'].median()
).astype(int)

train_idx, val_idx = train_test_split(range(len(smiles)), test_size=0.2, random_state=42)

smiles_train, smiles_val = smiles[train_idx], smiles[val_idx]
y_train, y_val = y[train_idx], y[val_idx]

train_loader = DataLoader(
    dataset=MolToGraph.create_pyg_data_lst(smiles_train, y_train),
    batch_size=32
)
val_loader = DataLoader(
    dataset=MolToGraph.create_pyg_data_lst(smiles_val, y_val),
    batch_size=32
)

_, result = models.train.train_graph_classifier(
    train_loader=train_loader, val_loader=val_loader,
    num_node_features=79, num_classes=2,
    model_name="GraphConv",
    c_hidden=256,
    layer_name="GraphConv",
    num_layers=3,
    dp_rate_linear=0.5,
    dp_rate=0.0,
    device=device,
    checkpoint_path=CHECKPOINT_PATH
)

print(result)

print('----------------- REGRESSION ---------------------')
CHECKPOINT_PATH = 'train_reg/'
y = solv['experimental value (kcal/mol)'].values

train_idx, val_idx = train_test_split(range(len(smiles)), test_size=0.2, random_state=42)

smiles_train, smiles_val = smiles[train_idx], smiles[val_idx]
y_train, y_val = y[train_idx], y[val_idx]

train_loader = DataLoader(
    dataset=MolToGraph.create_pyg_data_lst(smiles_train, y_train),
    batch_size=32
)
val_loader = DataLoader(
    dataset=MolToGraph.create_pyg_data_lst(smiles_val, y_val),
    batch_size=32
)

_, result = models.train.train_graph_regressor(
    train_loader=train_loader, val_loader=val_loader,
    num_node_features=79,
    model_name="GraphConv",
    c_hidden=256,
    layer_name="GraphConv",
    num_layers=3,
    dp_rate_linear=0.5,
    dp_rate=0.0,
    device=device,
    checkpoint_path=CHECKPOINT_PATH
)

print(result)
