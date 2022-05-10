import numpy as np
import torch
import torchvision.models as models
from torch import nn

from models import register
from .obs_transformer import HObservationTransformer
from .hsoftmax import HierarchicalSoftmax


@register
def resnet_supr(pretrained=False, **kwargs):
    return ResNetSelfSupr(pretrained, **kwargs)


@register
def resnet(pretrained=False, **kwargs):
    backbone = ResNetSelfSupr(pretrained, **kwargs)
    return ResNetClassifier(backbone, **kwargs)


@register
def hresnet50(pretrained=True, **kwargs):
    backbone = models.resnet50(pretrained=pretrained)
    return HierarchicalResNet(backbone=backbone, pretrained=pretrained, **kwargs)

@register
def obs_hresnet50(pretrained,  **kwargs):
    backbone = models.resnet50(pretrained=pretrained)
    return HObservationTransformer(encoder=backbone, **kwargs)

@register
def hresnet101(pretrained=True, **kwargs):
    backbone = models.resnet101(pretrained=pretrained)
    return HierarchicalResNet(backbone=backbone, pretrained=pretrained, **kwargs)


class ResNetSelfSupr(nn.Module):

    def __init__(self, pretrained, arch, num_classes):
        super(ResNetSelfSupr, self).__init__()
        self.backbone = models.resnet50(pretrained=pretrained, num_classes=num_classes)
        # self.backbone = self._get_basemodel(base_model)
        dim_mlp = self.backbone.fc.in_features

        # add mlp projection head
        self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)

    # def _get_basemodel(self, model_name):
    #     model = self.resnet_dict[model_name]
    #     return model

    def forward(self, x):
        return self.backbone(x)


class ResNetClassifier(nn.Module):

    def __init__(self, model: ResNetSelfSupr, feature_dim, num_classes: int):
        super(ResNetClassifier, self).__init__()
        self.model = model
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        x = self.model(x)
        return self.fc(x)

class HierarchicalResNet(nn.Module):

    def __init__(self, backbone, pretrained, num_classes, ntokens_per_class, **Kwargs):
        super(HierarchicalResNet, self).__init__()
        self.backbone = backbone

        # add mlp projection head
        dim_mlp = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(dim_mlp, 128)

        # dim_mlp = self.backbone.fc.out_features
        # self.fc = nn.Linear(dim_mlp, 128)
        self.hs = HierarchicalSoftmax(ntokens=num_classes, nhid=128, ntokens_per_class=ntokens_per_class)
    #     self.freeze()
    #
    # def freeze(self):
    #     for param in self.backbone.layer1.parameters():
    #         param.requires_grad = False
    #     for param in self.backbone.layer2.parameters():
    #         param.requires_grad = False

    def forward(self, x, y=None):
        x = self.backbone(x)
        x = nn.functional.relu(x)
        if y is None:
            return self.hs(x)
        # x = self.fc(x)
        loss, target_probs, layer_top_probs, layer_bottom_probs, top_indx, botton_indx, real_indx, preds = self.hs(x, y)
        return loss, real_indx, preds
