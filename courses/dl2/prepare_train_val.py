import numpy as np
import pandas as pd
from pathlib import Path

PATH = Path('data/carvana')
MASKS_FN = 'train_masks.csv'
META_FN = 'metadata.csv'
masks_csv = pd.read_csv(PATH/MASKS_FN)
meta_csv = pd.read_csv(PATH/META_FN)
TRAIN_DN = 'train-128'
MASKS_DN = 'train_masks-128'


def get_split(fold):
    file_names = [PATH/TRAIN_DN/o for o in masks_csv['img']]
    ids = np.array(list(set([f[:-7] for f in masks_csv['img']])))
    val_ids = np.random.choice(ids, 64, False)
    val_file_names = np.array([f for f in file_names if f.stem[:-3] in val_ids])
    train_file_names = np.array([f for f in file_names if f.stem[:-3] not in val_ids])
    
    return train_file_names, val_file_names