import torch.nn as nn

class MultiTaskHead(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.color_head = nn.Linear(in_features, 2)   # normal, bluish
        self.morph_head = nn.Linear(in_features, 3)  # clubbing, pitting, deformation

    def forward(self, x):
        color_logits = self.color_head(x)
        morph_logits = self.morph_head(x)
        return color_logits, morph_logits

