import torchvision.models as models
from torch import nn
from torchvision.models import EfficientNet

from .registry import register


@register
def efficientnet_b0(pretrained=False,  **kwargs):
    """
    arguments EfficientNet V1
    Args:
        dropout (float): The droupout probability, default 0.2
        stochastic_depth_prob (float): The stochastic depth probability
        num_classes (int): Number of classes
        last_channel (int): The number of channels on the penultimate layer
    """
    return models.efficientnet_b0(pretrained=pretrained, progress=True, kwargs=kwargs)
