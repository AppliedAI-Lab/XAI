"""
GNN encoder for fragment embedding. Same architecture as molecularGNN_3Dstructure
with forward_embed for use as feature extractor. Load pretrained weights from gnn_pretrain.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import spatial


class MolecularGNNEncoder(nn.Module):
    """Same as MolecularGraphNeuralNetwork but with forward_embed and no global device."""

    def __init__(self, N_atoms, dim, layer_hidden, layer_output):
        super(MolecularGNNEncoder, self).__init__()
        self.N_atoms = N_atoms
        self.dim = dim
        self.layer_hidden = layer_hidden
        self.layer_output = layer_output
        self.embed_atom = nn.Embedding(N_atoms, dim)
        self.gamma = nn.ModuleList([nn.Embedding(N_atoms, 1) for _ in range(layer_hidden)])
        self.W_atom = nn.ModuleList([nn.Linear(dim, dim) for _ in range(layer_hidden)])
        self.W_output = nn.ModuleList([nn.Linear(dim, dim) for _ in range(layer_output)])
        self.W_property = nn.Linear(dim, 1)

    def pad(self, matrices, pad_value, device):
        shapes = [m.shape for m in matrices]
        M, N = sum([s[0] for s in shapes]), sum([s[1] for s in shapes])
        zeros = torch.FloatTensor(np.zeros((M, N))).to(device)
        pad_matrices = pad_value + zeros
        i, j = 0, 0
        for k, matrix in enumerate(matrices):
            m, n = shapes[k]
            pad_matrices[i:i + m, j:j + n] = matrix
            i += m
            j += n
        return pad_matrices

    def update(self, matrix, vectors, layer):
        hidden_vectors = torch.relu(self.W_atom[layer](vectors))
        return hidden_vectors + torch.matmul(matrix, hidden_vectors)

    def sum_vectors(self, vectors, axis):
        sum_vectors = [torch.sum(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(sum_vectors)

    def forward(self, inputs):
        """Forward to property prediction (for pretraining)."""
        atoms, distance_matrices, molecular_sizes = inputs
        device = next(self.parameters()).device
        atoms = torch.cat(atoms)
        distance_matrix = self.pad(distance_matrices, 1e6, device)

        atom_vectors = self.embed_atom(atoms)
        for l in range(self.layer_hidden):
            gammas = torch.squeeze(self.gamma[l](atoms))
            M = torch.exp(-gammas * distance_matrix ** 2)
            atom_vectors = self.update(M, atom_vectors, l)
            atom_vectors = F.normalize(atom_vectors, 2, 1)

        for l in range(self.layer_output):
            atom_vectors = torch.relu(self.W_output[l](atom_vectors))

        molecular_vectors = self.sum_vectors(atom_vectors, molecular_sizes)
        properties = self.W_property(molecular_vectors)
        return properties

    def forward_embed(self, atoms, distance_matrices, molecular_sizes):
        """
        Forward to molecular embedding only (no property head). Use for fragment encoding.
        atoms: list of 1D LongTensor per molecule
        distance_matrices: list of 2D FloatTensor
        molecular_sizes: list of int (number of atoms per molecule)
        Returns: (batch, dim) tensor.
        """
        device = next(self.parameters()).device
        atoms_cat = torch.cat(atoms)
        distance_matrix = self.pad(distance_matrices, 1e6, device)

        atom_vectors = self.embed_atom(atoms_cat)
        for l in range(self.layer_hidden):
            gammas = torch.squeeze(self.gamma[l](atoms_cat))
            M = torch.exp(-gammas * distance_matrix ** 2)
            atom_vectors = self.update(M, atom_vectors, l)
            atom_vectors = F.normalize(atom_vectors, 2, 1)

        for l in range(self.layer_output):
            atom_vectors = torch.relu(self.W_output[l](atom_vectors))

        molecular_vectors = self.sum_vectors(atom_vectors, molecular_sizes)
        return molecular_vectors

    def forward_atom_embeddings(self, atoms, distance_matrices, molecular_sizes):
        """
        Per-atom embeddings after all GNN layers, before summing to molecular vector.
        atoms: list of 1D LongTensor per molecule (concatenated in model order)
        distance_matrices / molecular_sizes: same as forward_embed
        Returns: (total_atoms, dim) tensor in concatenation order.
        """
        device = next(self.parameters()).device
        atoms_cat = torch.cat(atoms)
        distance_matrix = self.pad(distance_matrices, 1e6, device)

        atom_vectors = self.embed_atom(atoms_cat)
        for l in range(self.layer_hidden):
            gammas = torch.squeeze(self.gamma[l](atoms_cat))
            M = torch.exp(-gammas * distance_matrix ** 2)
            atom_vectors = self.update(M, atom_vectors, l)
            atom_vectors = F.normalize(atom_vectors, 2, 1)

        for l in range(self.layer_output):
            atom_vectors = torch.relu(self.W_output[l](atom_vectors))
        return atom_vectors


def build_fragment_gnn_input(atoms_list, coords_list, atom_dict, device):
    """
    Build GNN input from list of (atoms, coords) per fragment.
    atoms_list: list of list of str (e.g. ['C','N','O'])
    coords_list: list of list of [x,y,z]
    atom_dict: dict str -> int (fixed mapping from pretrain)
    Returns (atoms_tensors, distance_matrices, sizes) for batching.
    """
    atoms_indices = []
    distance_matrices = []
    sizes = []
    for atoms, coords in zip(atoms_list, coords_list):
        if not atoms or not coords:
            continue
        idx = [atom_dict.get(a, 0) for a in atoms]
        atoms_indices.append(torch.LongTensor(idx).to(device))
        dist = spatial.distance_matrix(coords, coords)
        dist = np.where(dist == 0.0, 1e6, dist)
        distance_matrices.append(torch.FloatTensor(dist).to(device))
        sizes.append(len(atoms))
    return atoms_indices, distance_matrices, sizes


def load_pretrained_gnn(checkpoint_path, device=None):
    """
    Load pretrained GNN from checkpoint. Returns (model, atom_dict).
    checkpoint contains: state_dict, atom_dict, N_atoms, dim, layer_hidden, layer_output.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device)
    atom_dict = ckpt["atom_dict"]
    N_atoms = ckpt["N_atoms"]
    dim = ckpt["dim"]
    layer_hidden = ckpt["layer_hidden"]
    layer_output = ckpt["layer_output"]
    model = MolecularGNNEncoder(N_atoms, dim, layer_hidden, layer_output).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    return model, atom_dict
