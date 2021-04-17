import torch
import torch.nn as nn
import torch.nn.functional as F

class WoofNet(nn.Module):
    def __init__(self, backbone, n_classes, name='woofnet'):
        super(WoofNet, self).__init__()
        self.backbone = list(backbone.children())
        self.fc = nn.Linear(self.backbone[-1].in_features, n_classes)
        self.backbone = nn.Sequential(*self.backbone[:-1])
        self.name = name
    
    def forward(self, x):
        x = self.backbone(x).squeeze(-1).squeeze(-1)
        return self.fc(x)