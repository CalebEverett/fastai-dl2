import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from pathlib import Path

class CarvanaDataset(Dataset):
    def __init__(self, file_names: list, to_augment=False, transform=None, mode='train', problem_type=None):
        self.file_names = file_names
        self.to_augment = to_augment
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_file_name = Path(self.file_names[idx])
        img = load_image(img_file_name)
        mask = load_mask(img_file_name)

        img, mask = self.transform(img, mask)

        if self.mode == 'train':
            return to_float_tensor(img), to_float_tensor(mask)
        else:
            return to_float_tensor(img), str(img_file_name)


def to_float_tensor(img):
    return torch.from_numpy(np.moveaxis(img, -1, 0)).float()


def load_image(path):
    img = cv2.imread(str(path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_mask(path):
    mask_path = Path(path.parts[0], path.parts[1],
                     path.parts[-2].replace('-', '_masks-'), path.stem + '_mask.png')
    return cv2.imread(str(mask_path))