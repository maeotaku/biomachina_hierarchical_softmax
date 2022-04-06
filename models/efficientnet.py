import torchvision.models as models
from torch import nn
from torchvision.models import EfficientNet

from .registry import register


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
