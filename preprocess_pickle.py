import os
import pickle
import shutil
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
PICKLE_FILE = os.path.join(DATA_DIR, 'ch_2Dxysec.pickle')

TRAIN_FRAMES = 8000
VAL_FRAMES = 1000
TEST_FRAMES = 1000
TOTAL_FRAMES = TRAIN_FRAMES + VAL_FRAMES + TEST_FRAMES # 10000

def preprocess():
    if not os.path.exists(PICKLE_FILE):
        raise FileNotFoundError(f"Pickle file not found at {PICKLE_FILE}")

    print(f"Loading pickle data from {PICKLE_FILE} ...")
    with open(PICKLE_FILE, 'rb') as f:
        data = pickle.load(f)

    if not isinstance(data, np.ndarray):
        data = np.array(data)

    print(f"Loaded array shape: {data.shape}")
    if data.shape[0] != TOTAL_FRAMES:
        raise ValueError(f"Expected {TOTAL_FRAMES} frames, but got {data.shape[0]}")

    # Remove existing train/test directories to avoid any legacy files
    for folder in ['train', 'val', 'test']:
        folder_path = os.path.join(DATA_DIR, folder)
        if os.path.exists(folder_path):
            print(f"Removing existing folder {folder_path} ...")
            shutil.rmtree(folder_path)
        os.makedirs(folder_path, exist_ok=True)

    train_data = data[0:TRAIN_FRAMES]
    val_data = data[TRAIN_FRAMES:TRAIN_FRAMES+VAL_FRAMES]
    test_data = data[TRAIN_FRAMES+VAL_FRAMES:TOTAL_FRAMES]

    train_path = os.path.join(DATA_DIR, 'train', 'train_data.npy')
    val_path = os.path.join(DATA_DIR, 'val', 'val_data.npy')
    test_path = os.path.join(DATA_DIR, 'test', 'test_data.npy')

    print(f"Saving Train split -> {train_path}  (shape: {train_data.shape})")
    np.save(train_path, train_data)

    print(f"Saving Val split   -> {val_path}    (shape: {val_data.shape})")
    np.save(val_path, val_data)

    print(f"Saving Test split  -> {test_path}   (shape: {test_data.shape})")
    np.save(test_path, test_data)

    print("[OK] Preprocessing completed successfully.")

if __name__ == '__main__':
    preprocess()
