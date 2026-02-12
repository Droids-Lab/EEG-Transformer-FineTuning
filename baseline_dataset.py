import torch
from torch.utils.data import Dataset
import lmdb
import pickle

class EEG_LMDB_Dataset(Dataset):
    def __init__(self, lmdb_path, split="train"):
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False)
        with self.env.begin() as txn:
            self.keys = pickle.loads(txn.get(b'__keys__'))[split]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        with self.env.begin() as txn:
            data = pickle.loads(txn.get(self.keys[idx]))

        x = data['sample']        # (channels, time)
        y = data['label']         # 0 or 1

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)

        # optional normalization (recommended)
        x = (x - x.mean()) / (x.std() + 1e-6)

        return x, y
