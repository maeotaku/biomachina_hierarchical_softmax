import torchvision.models as models

from torch import nn

from models import register


@register
def resnet_supr(pretrained=False, **kwargs):
    return ResNetSelfSupr(pretrained, **kwargs)


@register
def resnet(pretrained=False, **kwargs):
    backbone = ResNetSelfSupr(pretrained, **kwargs)
    return ResNetClassifier(backbone, **kwargs)


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
