import lmdb
import pickle

lmdb_path = "processed_new/"

env = lmdb.open(lmdb_path, readonly=True, lock=False)
with env.begin() as txn:
    keys = pickle.loads(txn.get(b'__keys__'))

    print("Splits:", keys.keys())
    print("Number of train samples:", len(keys['train']))
    print("Number of test samples:", len(keys['test']))

    # Take ONE sample
    sample_key = keys['train'][0]
    data = pickle.loads(txn.get(sample_key))

    x = data['sample']
    y = data['label']

    print("Sample key:", sample_key)
    print("Sample shape:", x.shape)
    print("Sample dtype:", x.dtype)
    print("Label:", y)
