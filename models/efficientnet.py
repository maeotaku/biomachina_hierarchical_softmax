import torchvision.models as models
from torch import nn
from torchvision.models import EfficientNet

from .registry import register
from .resnet import HierarchicalSoftmax

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
    return HierarchicalEfficientNet(backbone=models.efficientnet_b0(pretrained=pretrained), **kwargs)

@register
def hefficientnet_b4(pretrained,  **kwargs):
    return HierarchicalEfficientNet(backbone=models.efficientnet_b4(pretrained=pretrained), **kwargs)


class HierarchicalEfficientNet(nn.Module):

    def __init__(self, backbone, num_classes, ntokens_per_class, **kwargs):
        super(HierarchicalEfficientNet, self).__init__()
        self.backbone = backbone

        # dim_mlp = self.backbone.classifier[1].in_features
        self.fc = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1000, 128),
            nn.ReLU(inplace=True)
        )
        self.hs = HierarchicalSoftmax(ntokens=num_classes, nhid=128, ntokens_per_class=ntokens_per_class)

    def forward(self, x, y):
        x = self.backbone(x)
        x = self.fc(x)
        loss, target_probs, layer_top_probs, layer_bottom_probs, top_indx, botton_indx, real_indx = self.hs(x, y)
        return loss, real_indx, target_probs
