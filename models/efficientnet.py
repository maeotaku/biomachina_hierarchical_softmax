import torchvision.models as models
from torch import nn
# from torchvision.models import EfficientNet

from .registry import register
from .hsoftmax import HierarchicalSoftmax
from efficientnet_pytorch import EfficientNet
from .obs_transformer import HObservationTransformer
import timm
import torch

@register
def efficientnet_b0(pretrained,  **kwargs):
    """
    arguments EfficientNet V1
    Args:
        pretrained (Boolean): define if want to download the pretrained weights.
        dropout (float): The droupout probability, default 0.2
        stochastic_depth_prob (float): The stochastic depth probability
        num_classes (int): Number of classes
        last_channel (int): The number of channels on the penultimate layer
    """
    return models.efficientnet_b0(pretrained=pretrained, progress=True, **kwargs)

@register
def efficientnet_b4(pretrained, num_classes, **kwargs):
    """
    arguments EfficientNet V1
    Args:
        pretrained (Boolean): define if want to download the pretrained weights.
        dropout (float): The droupout probability, default 0.2
        stochastic_depth_prob (float): The stochastic depth probability
        num_classes (int): Number of classes
        last_channel (int): The number of channels on the penultimate layer
    """
    model = models.efficientnet_b4(pretrained=pretrained, progress=True)
    model.classifier[1] = nn.Linear(in_features=1792, out_features=num_classes)
    return model


@register
def efficientnetv2_l(pretrained=False, **kwargs):

    class Effcientv2_l(nn.Module):

        def __init__(self, pretrained, num_classes, **kwargs):
            super(Effcientv2_l, self).__init__()
            module = timm.create_model('tf_efficientnet_b4', pretrained=pretrained, num_classes=num_classes)
            self.model = torch.compile(module)


        def forward(self, x, y):
            return self.model(x)

    return Effcientv2_l(pretrained, **kwargs)

register
def hefficientnet_b0(pretrained,  **kwargs):
    backbone = EfficientNet.from_pretrained('efficientnet-b0')
    # models.efficientnet_b4(pretrained=pretrained)
    return HierarchicalEfficientNet(backbone=backbone, **kwargs)

@register
def hefficientnet_b4(pretrained,  **kwargs):
    backbone = EfficientNet.from_pretrained('efficientnet-b4')
    # models.efficientnet_b4(pretrained=pretrained)
    return HierarchicalEfficientNet(backbone=backbone, **kwargs)

@register
def obs_hefficientnet_b4(pretrained,  **kwargs):
    backbone = EfficientNet.from_pretrained('efficientnet-b4')
    return HObservationTransformer(encoder=backbone, **kwargs)

@register
def selfefficientnet_b4(pretrained,  **kwargs):
    backbone = EfficientNet.from_pretrained('efficientnet-b4')
    return EfficientNetSelfSupr(backbone=backbone, **kwargs)

class HierarchicalEfficientNet(nn.Module):

    def __init__(self, backbone, num_classes, ntokens_per_class, **kwargs):
        super(HierarchicalEfficientNet, self).__init__()
        self.backbone = backbone
        self.backbone._fc = nn.Linear(self.backbone._fc.in_features, 128)
        self.hs = HierarchicalSoftmax(ntokens=num_classes, nhid=128, ntokens_per_class=ntokens_per_class)

    def forward(self, x, y=None):
        x = self.backbone(x)
        x = nn.functional.relu(x)
        if y is None:
            return self.hs(x)
        loss, target_probs, layer_top_probs, layer_bottom_probs, top_indx, botton_indx, real_indx, preds = self.hs(x, y)
        return loss, real_indx, preds

class EfficientNetSelfSupr(nn.Module):

    def __init__(self, backbone, **kwargs):
        super(EfficientNetSelfSupr, self).__init__()
        self.backbone = backbone
        self.backbone._fc = nn.Linear(self.backbone._fc.in_features, 256)

    def forward(self, x):
        x = self.backbone(x)
        return x
