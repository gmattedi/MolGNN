from numbers import Number
from typing import Sequence, Any, List

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from torch_geometric.data import Data


def one_hot_encoding(x: Any, values: Sequence[Any]) -> List[int]:
    """
    Sparse one-hot encoding of an input value, given a list of possible
    values. If x is not in values, an extra dimension is added to the vector and
    set to 1

    Args:
        x (Any)
        values (Sequence[Any]): Possible values

    Returns:
        binary_encoding (List[int]): Sparse one-hot vector

    """

    if x not in values:
        x = values[-1]

    binary_encoding = [int(v == x) for v in values]

    return binary_encoding


def get_atom_features(
        atom: Chem.Atom,
        use_chirality: bool = True,
        implicit_hydrogens: bool = True) -> np.ndarray:
    """
    Featurize atom
    
    Args:
        atom (Chem.Atom): Atom 
        use_chirality (bool) 
        implicit_hydrogens (bool)

    Returns:
        atom_feature_vector (np.ndarray)
    """

    allowed_elements = [
        'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I',
        'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'Li', 'Ge',
        'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown'
    ]

    if not implicit_hydrogens:
        allowed_elements = ['H'] + allowed_elements

    # compute atom features

    atom_type_enc = one_hot_encoding(str(atom.GetSymbol()), allowed_elements)

    n_heavy_neighbors_enc = one_hot_encoding(int(atom.GetDegree()), [0, 1, 2, 3, 4, "MoreThanFour"])

    formal_charge_enc = one_hot_encoding(int(atom.GetFormalCharge()), [-3, -2, -1, 0, 1, 2, 3, "Extreme"])

    hybridisation_type_enc = one_hot_encoding(str(atom.GetHybridization()),
                                              ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"])

    is_in_a_ring_enc = [int(atom.IsInRing())]

    is_aromatic_enc = [int(atom.GetIsAromatic())]

    atomic_mass_scaled = [float((atom.GetMass() - 10.812) / 116.092)]

    vdw_radius_scaled = [float((Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) - 1.5) / 0.6)]

    covalent_radius_scaled = [float((Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) - 0.64) / 0.76)]

    atom_feature_vector = \
        atom_type_enc + n_heavy_neighbors_enc + \
        formal_charge_enc + hybridisation_type_enc + \
        is_in_a_ring_enc + is_aromatic_enc + \
        atomic_mass_scaled + vdw_radius_scaled + \
        covalent_radius_scaled

    if use_chirality:
        chirality_type_enc = one_hot_encoding(str(atom.GetChiralTag()),
                                              ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW",
                                               "CHI_OTHER"])
        atom_feature_vector += chirality_type_enc

    if implicit_hydrogens:
        n_hydrogens_enc = one_hot_encoding(int(atom.GetTotalNumHs()), [0, 1, 2, 3, 4, "MoreThanFour"])
        atom_feature_vector += n_hydrogens_enc

    return np.array(atom_feature_vector)


def get_bond_features(bond: Chem.Bond,
                      use_stereochemistry: bool = True) -> np.ndarray:
    """
    Featurize bond

    Args:
        bond (Chem.Bond): Bond
        use_stereochemistry (bool)

    Returns:
        bond_feature_vector (np.ndarray)
    """

    allowed_bonds = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
                     Chem.rdchem.BondType.AROMATIC]

    bond_type_enc = one_hot_encoding(bond.GetBondType(), allowed_bonds)

    bond_is_conj_enc = [int(bond.GetIsConjugated())]

    bond_is_in_ring_enc = [int(bond.IsInRing())]

    bond_feature_vector = bond_type_enc + bond_is_conj_enc + bond_is_in_ring_enc

    if use_stereochemistry:
        stereo_type_enc = one_hot_encoding(str(bond.GetStereo()), ["STEREOZ", "STEREOE", "STEREOANY", "STEREONONE"])
        bond_feature_vector += stereo_type_enc

    return np.array(bond_feature_vector)


def create_pyg_data_lst(x_smiles: Sequence[str], y: Sequence[Number], device: str = 'cpu') -> List[Data]:
    """
    Package a sequence of smiles strings and labels as a list
    of PyTorch geometric data objects, containing the molecule as graph

    Args:
        x_smiles (Sequence[str])
        y (Sequence[Number])
        device (str)

    Returns:
        data_list (List[Data]): List of PyTorch geometric Data objects
    """

    # We use this hack to determine the number of edge and node features
    unrelated_smiles = "O=O"
    unrelated_mol = Chem.MolFromSmiles(unrelated_smiles)
    n_node_features = len(get_atom_features(unrelated_mol.GetAtomWithIdx(0)))
    n_edge_features = len(get_bond_features(unrelated_mol.GetBondBetweenAtoms(0, 1)))

    data_list = []

    for smiles, label in zip(x_smiles, y):

        # convert SMILES to RDKit mol object
        mol = Chem.MolFromSmiles(smiles)

        # get feature dimensions
        n_nodes = mol.GetNumAtoms()
        n_edges = 2 * mol.GetNumBonds()

        # construct node feature matrix X of shape (n_nodes, n_node_features)
        X = np.zeros((n_nodes, n_node_features))

        for atom in mol.GetAtoms():
            X[atom.GetIdx(), :] = get_atom_features(atom)

        X = torch.tensor(X, dtype=torch.float)

        # construct edge index array E of shape (2, n_edges)
        (rows, cols) = np.nonzero(GetAdjacencyMatrix(mol))
        torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
        torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)
        E = torch.stack([torch_rows, torch_cols], dim=0)

        # construct edge feature array EF of shape (n_edges, n_edge_features)
        EF = np.zeros((n_edges, n_edge_features))

        for (k, (i, j)) in enumerate(zip(rows, cols)):
            EF[k] = get_bond_features(mol.GetBondBetweenAtoms(int(i), int(j)))

        EF = torch.tensor(EF, dtype=torch.float)

        # construct label tensor
        y_tensor = torch.tensor(np.array([label]), dtype=torch.float)

        # construct Pytorch Geometric data object and append to data list
        data_list.append(Data(x=X, edge_index=E, edge_attr=EF, y=y_tensor).to(device))

    return data_list
