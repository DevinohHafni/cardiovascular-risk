import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from backbone import get_backbone
from multitask_head import MultiTaskHead
from dataset import NailDataset

dataset = NailDataset("data/raw")


# losses
color_loss_fn = nn.CrossEntropyLoss()
morph_loss_fn = nn.BCEWithLogitsLoss()

# model
backbone, feat_dim = get_backbone()
head = MultiTaskHead(feat_dim)

model = nn.Sequential(backbone, head)

