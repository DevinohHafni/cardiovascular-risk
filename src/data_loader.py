import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold

def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
    ])

def load_dataset(data_dir="data/raw"):
    dataset = datasets.ImageFolder(
        root=data_dir,
        transform=get_transforms()
    )
    return dataset

def get_kfold_loaders(dataset, k=5, batch_size=16):
    targets = np.array(dataset.targets)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(np.zeros(len(targets)), targets)
    ):
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(
            train_subset, batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_subset, batch_size=batch_size, shuffle=False
        )

        yield fold, train_loader, val_loader
