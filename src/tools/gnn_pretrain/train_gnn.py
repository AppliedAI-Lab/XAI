"""
Train GNN on QM9 (molecularGNN_3Dstructure dataset) and save weights for use as
fragment encoder in the attention pipeline. Run once; then load weights in attn without training.
"""
import os
import sys
import timeit
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict
from scipy import spatial
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.models.gnn_encoder import MolecularGNNEncoder

# Dataset path relative to this repo
DATASET_DIR = "/mnt/time-series/bio/XAI/SubstructureAttn/src/models/molecularGNN_3Dstructure/dataset"


def create_atoms(atoms, atom_dict):
    atoms = [atom_dict[a] for a in atoms]
    return np.array(atoms)


def create_distances(coords):
    distance_matrix = spatial.distance_matrix(coords, coords)
    return np.where(distance_matrix == 0.0, 1e6, distance_matrix)


def split_dataset(dataset, ratio):
    np.random.seed(1234)
    np.random.shuffle(dataset)
    n = int(ratio * len(dataset))
    return dataset[:n], dataset[n:]


def create_datasets(dataset_name, physical_property, device):
    dir_dataset = os.path.join(DATASET_DIR, dataset_name, "")
    if not os.path.isdir(dir_dataset):
        raise FileNotFoundError("Dataset not found: " + dir_dataset)

    atom_dict = defaultdict(lambda: len(atom_dict))

    def load_file(filename):
        filepath = os.path.join(dir_dataset, filename)
        with open(filepath, "r") as f:
            property_types = f.readline().strip().split()
            data_original = f.read().strip().split("\n\n")
        property_index = property_types.index(physical_property)
        out = []
        for data in data_original:
            data = data.strip().split("\n")
            prop_val = float(data[-1].split()[property_index])
            atoms, atom_coords = [], []
            for atom_xyz in data[1:-1]:
                atom, x, y, z = atom_xyz.split()
                atoms.append(atom)
                atom_coords.append([float(x), float(y), float(z)])
            atoms = create_atoms(atoms, atom_dict)
            distance_matrix = create_distances(atom_coords)
            molecular_size = len(atoms)
            atoms_t = torch.LongTensor(atoms).to(device)
            dist_t = torch.FloatTensor(distance_matrix).to(device)
            prop_t = torch.FloatTensor([[prop_val]]).to(device)
            out.append((atoms_t, dist_t, molecular_size, prop_t))
        return out

    dataset_train = load_file("data_train.txt")
    dataset_train, dataset_dev = split_dataset(dataset_train, 0.9)
    dataset_test = load_file("data_test.txt")
    N_atoms = len(atom_dict)
    return dataset_train, dataset_dev, dataset_test, N_atoms, dict(atom_dict)


def train_epoch(model, dataset, optimizer, batch_train, device, epoch):
    np.random.shuffle(dataset)
    loss_total = 0
    N = len(dataset)
    for i in tqdm(range(0, N, batch_train), desc=f"Train epoch {epoch}", leave=False):
        data_batch = list(zip(*dataset[i : i + batch_train]))
        inputs = data_batch[:-1]
        correct = torch.cat(data_batch[-1])
        pred = model(inputs)
        loss = F.mse_loss(pred, correct)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_total += loss.item()
    return loss_total


def test_mae(model, dataset, batch_test):
    N = len(dataset)
    SAE = 0
    for i in range(0, N, batch_test):
        data_batch = list(zip(*dataset[i : i + batch_test]))
        inputs = data_batch[:-1]
        correct = torch.cat(data_batch[-1])
        with torch.no_grad():
            pred = model(inputs)
        SAE += torch.abs(pred - correct).sum().item()
    return SAE / N


def main(
    dataset_name="QM9_under14atoms",
    property_name="U0(kcalmol^-1)",
    dim=200,
    layer_hidden=6,
    layer_output=6,
    batch_train=32,
    batch_test=32,
    lr=1e-3,
    lr_decay=0.99,
    decay_interval=10,
    iteration=9000,
    save_path=None,
):
    if save_path is None:
        os.makedirs(os.path.join(ROOT, "weights"), exist_ok=True)
        save_path = os.path.join(ROOT, "weights", "gnn_pretrain.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    print("Loading dataset", dataset_name)
    dataset_train, dataset_dev, dataset_test, N_atoms, atom_dict = create_datasets(
        dataset_name, property_name, device
    )
    print("Train:", len(dataset_train), "Dev:", len(dataset_dev), "Test:", len(dataset_test), "N_atoms:", N_atoms)

    torch.manual_seed(1234)
    model = MolecularGNNEncoder(N_atoms, dim, layer_hidden, layer_output).to(device)
    for i in range(layer_hidden):
        model.gamma[i].weight.data = torch.ones((N_atoms, 1)).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("Training for", iteration, "epochs")
    start = timeit.default_timer()
    best_dev = float("inf")
    best_epoch = None
    for epoch in range(1, iteration + 1):
        if epoch % decay_interval == 0:
            for g in optimizer.param_groups:
                g["lr"] *= lr_decay
        loss_train = train_epoch(model, dataset_train, optimizer, batch_train, device, epoch)
        elapsed = timeit.default_timer() - start
        if epoch % 100 == 0 or epoch == 1:
            mae_dev = test_mae(model, dataset_dev, batch_test)
            mae_test = test_mae(model, dataset_test, batch_test)
            print(epoch, elapsed, loss_train, mae_dev, mae_test)
            if mae_dev < best_dev:
                best_dev = mae_dev
                best_epoch = epoch
                checkpoint = {
                    "state_dict": model.state_dict(),
                    "atom_dict": atom_dict,
                    "N_atoms": N_atoms,
                    "dim": dim,
                    "layer_hidden": layer_hidden,
                    "layer_output": layer_output,
                }
                torch.save(checkpoint, save_path)
                print(f"Saved best checkpoint at epoch {epoch} with dev MAE {mae_dev} to {save_path}")

    if best_epoch is not None:
        print(f"Best dev MAE {best_dev} at epoch {best_epoch}. Checkpoint saved to {save_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="QM9_under14atoms")
    p.add_argument("--property", default="U0(kcalmol^-1)")
    p.add_argument("--dim", type=int, default=200)
    p.add_argument("--layer_hidden", type=int, default=6)
    p.add_argument("--layer_output", type=int, default=6)
    p.add_argument("--batch_train", type=int, default=32)
    p.add_argument("--batch_test", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr_decay", type=float, default=0.99)
    p.add_argument("--decay_interval", type=int, default=10)
    p.add_argument("--iteration", type=int, default=3000)
    p.add_argument("--save_path", default=None)
    args = p.parse_args()
    main(
        dataset_name=args.dataset,
        property_name=args.property,
        dim=args.dim,
        layer_hidden=args.layer_hidden,
        layer_output=args.layer_output,
        batch_train=args.batch_train,
        batch_test=args.batch_test,
        lr=args.lr,
        lr_decay=args.lr_decay,
        decay_interval=args.decay_interval,
        iteration=args.iteration,
        save_path=args.save_path,
    )
