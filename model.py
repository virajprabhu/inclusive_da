import sys
import torch
import torch.nn as nn
from torch.nn import init
import torchvision
import torch.nn.functional as F
from torchvision import models

import numpy as np

# from utils import ReverseLayerF

np.random.seed(1234)
torch.manual_seed(1234)


class TaskNet(nn.Module):
    "Basic class which does classification."

    def __init__(self, num_cls=10, l2_normalize=False, temperature=1.0):
        super(TaskNet, self).__init__()
        self.num_cls = num_cls
        self.l2_normalize = l2_normalize
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

        dim, pred_dim = 2048, 512
        self.proj_layer = nn.Sequential(
            nn.Linear(dim, pred_dim, bias=False),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(inplace=True),  # hidden layer
            nn.Linear(pred_dim, dim),
        )  # output layer

    def forward(self, x, reverse_grad=False, get_feats=False, get_proj=False):
        # Extract features
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        x = x.clone()
        feats = self.fc_params(x)

        # Classify
        if self.l2_normalize:
            feats = F.normalize(feats)
        score = self.classifier(feats) / self.temperature

        if get_feats:
            if get_proj:
                proj = self.proj_layer(feats)
                return score, feats, proj.detach()
            else:
                return score, feats
        else:
            return score

    def load(self, init_path):
        net_init_dict = torch.load(init_path, map_location=torch.device("cpu"))
        self.load_state_dict(net_init_dict, strict=False)

    def save(self, out_path):
        torch.save(self.state_dict(), out_path)


class ResNet50(TaskNet):
    def __init__(
        self, num_cls=10, l2_normalize=False, temperature=1.0, pretrained=True
    ):
        super(ResNet50, self).__init__(num_cls, l2_normalize, temperature)
        model = models.resnet50(pretrained=pretrained)
        model.fc = nn.Identity()
        self.conv_params = model
        self.fc_params = nn.Identity()
        bias = False if self.l2_normalize else True
        self.classifier = nn.Linear(2048, self.num_cls, bias=bias)

        init.xavier_normal_(self.classifier.weight)
        if bias:
            self.classifier.bias.data.zero_()
