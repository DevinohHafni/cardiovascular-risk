import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class NailDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.samples = []
        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])

        for folder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder)
            if not os.path.isdir(folder_path):
                continue

            for img in os.listdir(folder_path):
                self.samples.append((folder, os.path.join(folder_path, img)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        folder, img_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # Task 1: Color label
        if folder == "Healthy_Nail":
            color = 0  # normal
        elif folder == "blue_finger":
            color = 1  # bluish
        else:
            color = -1  # unknown / ignore

        # Task 2: Morphology (multi-label)
        clubbing = 1 if folder == "clubbing" else 0
        pitting = 1 if folder == "pitting" else 0
        deformation = 1 if folder == "Onychogryphosis" else 0

        morph = torch.tensor([clubbing, pitting, deformation]).float()

        return image, color, morph
