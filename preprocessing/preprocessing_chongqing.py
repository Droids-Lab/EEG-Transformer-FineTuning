import os
import lmdb
import mne
import numpy as np
import pickle
import random

root_dir = 'CHONGQING/raw_eeg'

# Labels: 
# 0 = Low ASD (score 4-6) - 13 subjects 
# 1 = High ASD (score 7-10) - 10 subjects

# labels = {
#     'QJL2024029MMN': 0,
#     'QJL2024001MMN': 0,
#     'QJL2024003MMN': 1,
#     'QJL2024004MMN': 1,
#     'QJL2024007MMN': 0,
#     'QJL2024011MMN': 1,
#     'QJL2024012MMN': 0,
#     'QJL2024014MMN': 1,
#     'QJL2024015MMN': 0,
#     'QJL2024016MMN': 1,
#     'QJL2024017MMN': 0,
#     'QJL2024020MMN': 0,
#     'QJL2024021MMN': 0,
#     'QJL2024024MMN': 0,
#     'QJL2024031MMN': 0,
#     'QJL2024047MMN': 1,
#     'QJL2024103MMN': 1,
#     'TY25001MMN': 1,
#     'TY25002MMN': 0,
#     'TY25003MMN': 0,
#     'TY25004MMN': 0,
#     'TY25010MMN': 1,
#     'TY25011MMN': 1
# }

labels = {
    'QJL2024029MMN': 6,
    'QJL2024001MMN': 6,
    'QJL2024003MMN': 7,
    'QJL2024004MMN': 7,
    'QJL2024007MMN': 5,
    'QJL2024011MMN': 10,
    'QJL2024012MMN': 4,
    'QJL2024014MMN': 8,
    'QJL2024015MMN': 5,
    'QJL2024016MMN': 8,
    'QJL2024017MMN': 4,
    'QJL2024020MMN': 6,
    'QJL2024021MMN': 6,
    'QJL2024024MMN': 4,
    'QJL2024031MMN': 6,
    'QJL2024047MMN': 8,
    'QJL2024103MMN': 8,
    'TY25001MMN': 8,
    'TY25002MMN': 6,
    'TY25003MMN': 6,
    'TY25004MMN': 4,
    'TY25010MMN': 8,
    'TY25011MMN': 7
}

files = [file for file in os.listdir(root_dir)]
files = sorted(files)

# Resampling details: 
TARGET_SF = 200 # Sampling frequency
SEG_SECONDS = 30 # 30 Second segments
SEG_LEN = SEG_SECONDS * TARGET_SF # Total Segment length

db = lmdb.open(r'processed_new', map_size=5*1024*1024*1024) #overestimate of maximum database size (5 GB)
dataset_keys = {'train': [], 'val': [], 'test': []}  # Optional, track keys per split

subject_ids = list(labels.keys())
random.shuffle(subject_ids)
print("shuffle results: ", subject_ids)

split = {
    'train': subject_ids[:16],
    'test': subject_ids[16:]
}

# Loop through each subject folder
for subject_id in os.listdir(root_dir):
    subject_path = os.path.join(root_dir, subject_id)
    
    if subject_id not in labels:
        print(f"Skipping {subject_id} (no label assigned)")
        continue

    # Find the .vhdr file inside this folder
    for file in os.listdir(subject_path):
        if file.endswith(".vhdr"):
            vhdr_path = os.path.join(subject_path, file)
            print("Loading:", vhdr_path)
            
            # Load the BrainVision EEG
            raw = mne.io.read_raw_brainvision(vhdr_path, preload=True)
            # Keep frequencies between 0.3 Hz and 75 Hz
            raw.filter(l_freq=0.3, h_freq=75)
            # Remove line noise at 50 Hz
            raw.notch_filter(freqs=50)

            raw.resample(TARGET_SF) # Resample to 200 Hz 

            # convert to numpy shape : (n_channels, n_times)
            eeg_uv = raw.get_data() * 1e6 # convert from V to µV
            n_channels, n_points = eeg_uv.shape # n_points : total number of time samples recorded after resampling
                                                # i.e : sampling rate (200) * total recording time
            n_windows = n_points // SEG_LEN # the number of full 30 second windows
            print("Number of 30-second windows:", n_windows)
            if n_windows == 0:
                # skip recording if it is too short (less than 30 seconds of EEG) 
                print("EEG too short")
                pass
            eeg_uv_trim = eeg_uv[:, :n_windows * SEG_LEN] # trims leftover data
            # format still follows eeg_uv : [n_channels, n_windows*SEG_LEN]

            # reshape to windows: (n_windows, n_channels, SEG_LEN)
            windows = eeg_uv_trim.reshape(n_channels, n_windows, SEG_LEN).transpose(1, 0, 2)
            # .reshape: slices continuous EEG data into n_windows (number of full windows) windows, each SEG_LEN (6000) samples long
            # .transpose: just changes the dimensions to be more natural for machine learning - no. samples first

            # now break each 30s window into 30 sub-windows of 200 samples each:
            windows_30x200 = windows.reshape(n_windows, n_channels, 30, 200)

            label = labels[subject_id]
            print(f"{subject_id}: label={label}, EEG shape={windows_30x200.shape}")

            for i, window in enumerate(windows_30x200): # Loop over each 30-second window
                for j in range(window.shape[1]):  # Loop over sub-windows within that window
                    sub_window = window[:, j, :]
                    key = f"{subject_id}-{i}-{j}".encode() # Create a unique identifier for this sub-window.
                                                            # Example: "QJL2024001MMN-0-5" → subject QJL2024001MMN, first 30-second window
                    data_dict = {'sample': sub_window, 'label': label} # Store both the EEG data and its label in a dictionary.

                    txn = db.begin(write=True)
                    txn.put(key=key, value=pickle.dumps(data_dict)) # Each sub-window is one entry in the database.
                    txn.commit()

                    # Keep track of keys per split
                    for split_name, split_subjects in split.items():
                        if subject_id in split_subjects:
                            dataset_keys[split_name].append(key)

# Save the list of keys
txn = db.begin(write=True)
txn.put(b'__keys__', pickle.dumps(dataset_keys))
txn.commit()
db.close()
print("LMDB database creation complete!")
