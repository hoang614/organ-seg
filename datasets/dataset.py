from monai.data import Dataset
from .text_utils import read_text_file
from .data_paths import build_dataset_paths
import random

def build_train_dataset(img_paths, label_paths, txt_paths, transforms):
    train_files = [
        {
            "image": img,
            "label": label,
            "text": read_text_file(txt)
        }
        for img, label, txt in zip(img_paths, label_paths, txt_paths)
    ]

    random.shuffle(train_files)

    return Dataset(
        data=train_files,
        transform=transforms
    )