import numpy as np
import torch
import torchvision.models as models
from torch import nn

from models import register


@register
def hresnet(pretrained=False, **kwargs):
    return HierarchicalResNet(pretrained, **kwargs)


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


class HierarchicalSoftmax(nn.Module):

    def __init__(self, ntokens, nhid, ntokens_per_class=None, **kwargs):
        super(HierarchicalSoftmax, self).__init__()

        # Parameters
        self.ntokens = ntokens
        self.nhid = nhid

        if ntokens_per_class is None:
            ntokens_per_class = int(np.ceil(np.sqrt(ntokens)))

        self.ntokens_per_class = ntokens_per_class

        self.nclasses = int(np.ceil(self.ntokens * 1. / self.ntokens_per_class))
        self.ntokens_actual = self.nclasses * self.ntokens_per_class

        self.layer_top_W = nn.Parameter(torch.FloatTensor(self.nhid, self.nclasses), requires_grad=True)
        # print(self.layer_top_W.shape)
        self.layer_top_b = nn.Parameter(torch.FloatTensor(self.nclasses), requires_grad=True)

        self.layer_bottom_W = nn.Parameter(torch.FloatTensor(self.nclasses, self.nhid, self.ntokens_per_class),
                                           requires_grad=True)
        # print(self.layer_bottom_W.shape)
        self.layer_bottom_b = nn.Parameter(torch.FloatTensor(self.nclasses, self.ntokens_per_class), requires_grad=True)

        self.softmax = nn.Softmax(dim=1)

        self.init_weights()



    def init_weights(self):
        initrange = 0.1
        self.layer_top_W.data.uniform_(-initrange, initrange)
        self.layer_top_b.data.fill_(0)
        self.layer_bottom_W.data.uniform_(-initrange, initrange)
        self.layer_bottom_b.data.fill_(0)

    def forward(self, inputs, labels):
        batch_size, d = inputs.size()

        label_position_top = (labels / self.ntokens_per_class).long()
        label_position_bottom = (labels % self.ntokens_per_class).long()

        layer_top_logits = torch.matmul(inputs, self.layer_top_W) + self.layer_top_b
        layer_top_probs = self.softmax(layer_top_logits)

        layer_bottom_logits = torch.squeeze(
            torch.bmm(torch.unsqueeze(inputs, dim=1), self.layer_bottom_W[label_position_top]), dim=1) + \
                              self.layer_bottom_b[label_position_top]
        layer_bottom_probs = self.softmax(layer_bottom_logits)

        target_probs = layer_top_probs[torch.arange(batch_size).long(), label_position_top] * layer_bottom_probs[
            torch.arange(batch_size).long(), label_position_bottom]

        # print(f"top {layer_top_probs.shape} {layer_top_probs}")
        # print(f"bottom {layer_bottom_probs.shape} {layer_bottom_probs}")
        top_indx = torch.argmax(layer_top_probs, dim=1)
        botton_indx = torch.argmax(layer_bottom_probs, dim=1)

        real_indx = (top_indx * self.ntokens_per_class) + botton_indx
        # print(top_indx, self.nclasses, botton_indx)
        # print(f"target {target_probs.shape} {target_probs}")

        # loss = -torch.mean(torch.log(target_probs.type(torch.float32) + 1e-3))
        loss = -torch.mean(torch.log(target_probs))

        return loss, target_probs, layer_top_probs, layer_bottom_probs, top_indx, botton_indx, real_indx


class HierarchicalResNet(nn.Module):

    def __init__(self, pretrained, num_classes, ntokens_per_class, **Kwargs):
        super(HierarchicalResNet, self).__init__()
        self.backbone = models.resnet50(pretrained=pretrained)  # , num_classes=out_dim)

        # add mlp projection head
        dim_mlp = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(dim_mlp, 128)

        # dim_mlp = self.backbone.fc.out_features
        # self.fc = nn.Linear(dim_mlp, 128)
        self.hs = HierarchicalSoftmax(ntokens=num_classes, nhid=128, ntokens_per_class=ntokens_per_class)
        self.freeze()

    def freeze(self):
        for param in self.backbone.layer1.parameters():
            param.requires_grad = False
        for param in self.backbone.layer2.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        x = self.backbone(x)
        # x = self.fc(x)
        loss, target_probs, layer_top_probs, layer_bottom_probs, top_indx, botton_indx, real_indx = self.hs(x, y)
        return loss, real_indx
