import numpy as np
import torch
import torchvision.models as models
from torch import nn

from models import register
from .resnet import HierarchicalSoftmax


@register
def hdensenet(pretrained=False, **kwargs):
    return HierarchicalDenseNet(pretrained, **kwargs)

class HierarchicalDenseNet(nn.Module):

    def __init__(self, pretrained, num_classes, ntokens_per_class, **Kwargs):
        super(HierarchicalDenseNet, self).__init__()
        self.backbone = models.densenet161(pretrained=pretrained)  # , num_classes=out_dim)

        dim_mlp = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Linear(dim_mlp, 128)

        self.hs = HierarchicalSoftmax(ntokens=num_classes, nhid=128, ntokens_per_class=ntokens_per_class)

    def forward(self, x, y):
        x = self.backbone(x)
        loss, target_probs, layer_top_probs, layer_bottom_probs, top_indx, botton_indx, real_indx = self.hs(x, y)
        return loss, real_indx
