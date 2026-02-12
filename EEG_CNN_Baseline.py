import os
import pickle
import lmdb
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ------------------------
# Dataset
# ------------------------
class EEG_LMDB_Dataset(Dataset):
    def __init__(self, lmdb_path, split="train"):
        super().__init__()
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=True, meminit=False)
        with self.env.begin(write=False) as txn:
            all_keys = pickle.loads(txn.get(b'__keys__'))
            self.keys = all_keys[split]

        # automatically get sample shape
        with self.env.begin(write=False) as txn:
            sample = pickle.loads(txn.get(self.keys[0]))['sample']
            self.n_channels, self.n_timepoints = sample.shape

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        with self.env.begin(write=False) as txn:
            data = pickle.loads(txn.get(key))

        x = torch.tensor(data['sample'], dtype=torch.float32)
        y = torch.tensor(data['label'], dtype=torch.long)

        # per-sample normalization
        x = (x - x.mean()) / (x.std() + 1e-6)
        return x, y

    # optional collate_fn for DataLoader
    def collate(self, batch):
        xs = torch.stack([b[0] for b in batch])
        ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
        return xs, ys

# ------------------------
# CNN Baseline Model
# ------------------------
class EEG_CNN_Baseline(nn.Module):
    def __init__(self, n_channels, n_timepoints, n_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 7), padding=(1, 3))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 7), padding=(1, 3))
        self.pool = nn.MaxPool2d((2, 2))
        self.dropout = nn.Dropout(0.3)

        # calculate flattened feature dimension
        self.feature_dim = 32 * (n_channels // 4) * (n_timepoints // 4)

        self.fc1 = nn.Linear(self.feature_dim, 128)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        # x shape: (B, C, T)
        x = x.unsqueeze(1)  # (B, 1, C, T)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.flatten(start_dim=1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# ------------------------
# Training & Evaluation
# ------------------------
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return total_loss / len(loader), correct / total

def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item()
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    return total_loss / len(loader), correct / total

# ------------------------
# Main Function
# ------------------------
def main():
    import random
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lmdb_path = "processed_new/"

    # ------------------------
    # Dataset & DataLoader
    # ------------------------
    train_set = EEG_LMDB_Dataset(lmdb_path, split="train")
    test_set  = EEG_LMDB_Dataset(lmdb_path, split="test")

    print("Train class distribution:", Counter([train_set[i][1].item() for i in range(len(train_set))]))
    print("Test class distribution:", Counter([test_set[i][1].item() for i in range(len(test_set))]))

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, collate_fn=train_set.collate, num_workers=0)
    test_loader  = DataLoader(test_set, batch_size=64, shuffle=False, collate_fn=test_set.collate, num_workers=0)

    # ------------------------
    # Model, Loss, Optimizer
    # ------------------------
    model = EEG_CNN_Baseline(
        n_channels=train_set.n_channels,
        n_timepoints=train_set.n_timepoints,
        n_classes=2
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # ------------------------
    # Training Loop
    # ------------------------
    EPOCHS = 30
    for epoch in range(EPOCHS):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        te_loss, te_acc = eval_one_epoch(model, test_loader, criterion, device)
        print(f"Epoch [{epoch+1}/{EPOCHS}] | "
              f"Train Acc: {tr_acc:.4f} | Test Acc: {te_acc:.4f}")

if __name__ == "__main__":
    main()
