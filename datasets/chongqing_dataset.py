import os
import lmdb
import pickle
import numpy as np
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import torch

# Utility to convert numpy arrays to torch tensors
def to_tensor(array):
    return torch.from_numpy(np.array(array, dtype=np.float32))

class CustomDataset(Dataset):
    def __init__(self, lmdb_dir, mode='train'):
        super(CustomDataset, self).__init__()
        self.db = lmdb.open(lmdb_dir, readonly=True, lock=False, readahead=True, meminit=False)
        with self.db.begin(write=False) as txn:
            # __keys__ is a dict: {'train': [...], 'test': [...]}
            all_keys = pickle.loads(txn.get('__keys__'.encode()))
            self.keys = all_keys[mode]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        with self.db.begin(write=False) as txn:
            pair = pickle.loads(txn.get(key))
        x = pair['sample']
        y = pair['label']
        x = np.array(x, dtype=np.float32)
        y = np.array(y, dtype=np.int64)
        return to_tensor(x), torch.tensor(y)

    def collate(self, batch):
        x_data = np.array([x[0] for x in batch])
        y_label = np.array([x[1] for x in batch])
        return to_tensor(x_data), torch.tensor(y_label).long()

class LoadDataset:
    def __init__(self, params):
        # self.datasets_dir = params.datasets_dir
        self.datasets_dir = "processed_new/"
        self.batch_size = params.batch_size
        # self.num_workers = params.num_workers
        self.num_workers = 16

    def get_data_loader(self):
        train_set = CustomDataset(self.datasets_dir, mode='train')
        test_set = CustomDataset(self.datasets_dir, mode='test')

        print("Train class distribution:", Counter([train_set[i][1].item() for i in range(len(train_set))]))
        print("Test class distribution:", Counter([test_set[i][1].item() for i in range(len(test_set))]))

        data_loader = {
            'train': DataLoader(
                train_set,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=train_set.collate,
                num_workers=0
            ),
            'test': DataLoader(
                test_set,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=test_set.collate,
                num_workers=0
            ),
        }
        return data_loader

# Optional: helper to rebuild __keys__ with train/test split
def rebuild_keys(lmdb_dir, train_ratio=0.8):
    db = lmdb.open(lmdb_dir, readonly=False, lock=False)
    with db.begin(write=True) as txn:
        all_data_keys = [key.decode() for key in txn.cursor().iternext(values=False) if key.decode() != '__keys__']
        all_labels = [pickle.loads(txn.get(k.encode()))['label'] for k in all_data_keys]

        # stratified split
        keys_class0 = [k for k, label in zip(all_data_keys, all_labels) if label == 0]
        keys_class1 = [k for k, label in zip(all_data_keys, all_labels) if label == 1]

        train0 = keys_class0[:int(len(keys_class0)*train_ratio)]
        test0 = keys_class0[int(len(keys_class0)*train_ratio):]

        train1 = keys_class1[:int(len(keys_class1)*train_ratio)]
        test1 = keys_class1[int(len(keys_class1)*train_ratio):]

        keys_dict = {
            'train': train0 + train1,
            'test': test0 + test1
        }

        txn.put('__keys__'.encode(), pickle.dumps(keys_dict))
    print("LMDB keys rebuilt with train/test split.")
