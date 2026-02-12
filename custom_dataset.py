
import os
import numpy as np
from glob import glob
from PIL import Image
from torch.utils.data import Dataset

class BlockLevelDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.classes = ["anterior", "middle", "posterior"]
        self.class_map = {
            "block_1": "anterior",
            "block_2": "middle",
            "block_3": "posterior"
        }

        all_imgs = glob(os.path.join(root_dir, "**", "*.jpg"), recursive=True)
        for img_path in all_imgs:
            label = self.extract_class(img_path)
            if label is not None:
                self.samples.append((img_path, self.classes.index(label)))

    def extract_class(self, path):
        parts = path.lower().split(os.sep)
        for part in parts:
            if "block_1" in part:
                return "anterior"
            elif "block_2" in part:
                return "middle"
            elif "block_3" in part:
                return "posterior"
        return None

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(image=np.array(img))["image"]
        return img, label

    def __len__(self):
        return len(self.samples)
