import torchvision.models as models
from torch import nn
# from torchvision.models import EfficientNet

from .registry import register
from .resnet import HierarchicalSoftmax
from efficientnet_pytorch import EfficientNet

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
def selfefficientnet_b4(pretrained,  **kwargs):
    backbone = EfficientNet.from_pretrained('efficientnet-b4')
    # models.efficientnet_b4(pretrained=pretrained)
    return EfficientNetSelfSupr(backbone=backbone, **kwargs)


class HierarchicalEfficientNet(nn.Module):

    def __init__(self, backbone, num_classes, ntokens_per_class, **kwargs):
        super(HierarchicalEfficientNet, self).__init__()
        self.backbone = backbone

        # dim_mlp = self.backbone.classifier[1].in_features
        self.backbone._fc = nn.Linear(self.backbone._fc.in_features, 128)
        self.hs = HierarchicalSoftmax(ntokens=num_classes, nhid=128, ntokens_per_class=ntokens_per_class)

    def forward(self, x, y):
        x = self.backbone(x)
        loss, target_probs, layer_top_probs, layer_bottom_probs, top_indx, botton_indx, real_indx, preds = self.hs(x, y)
        return loss, real_indx, preds

class EfficientNetSelfSupr(nn.Module):

    def __init__(self, backbone, **kwargs):
        super(EfficientNetSelfSupr, self).__init__()
        self.backbone = backbone

    def forward(self, x):
        x = self.backbone(x)
        return x
