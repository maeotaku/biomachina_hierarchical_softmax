import torch
from einops import repeat, rearrange
from einops.layers.torch import Rearrange
from torch import nn

from models import register
from .hsoftmax import HierarchicalSoftmax


@register
def vit_encoder(pretrained=False, **kwargs):
    return ViTEncoder(**kwargs)


import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


@register
def timm_model(pretrained=False, pretrained_version="vit_base_patch16_224", **kwargs):

    class TIMM(nn.Module):

        def __init__(self, pretrained, pretrained_version, num_classes, **kwargs):
            super(TIMM, self).__init__()
            self.model = timm.create_model(pretrained_version, pretrained=pretrained, num_classes=num_classes)

        def change_head(self, num_classes):
            self.model.head = nn.Linear(
                in_features=self.model.head.in_features,
                out_features=num_classes
            )

        def forward(self, x, y):
            return self.model(x)

    return TIMM(pretrained, pretrained_version, **kwargs)




# @register
# def htimm_model(pretrained=False, pretrained_version="vit_base_patch16_224", **kwargs):

#     class TIMM(nn.Module):

#         def __init__(self, pretrained, pretrained_version, num_classes, ntokens_per_class, **kwargs):
#             super(TIMM, self).__init__()
#             self.model = timm.create_model(pretrained_version, pretrained=pretrained) #, num_classes=256)
#             self.model.global_pool == 'avg'
#             self.linear = nn.Linear(1000, 256)
#             self.hs = HierarchicalSoftmax(ntokens=num_classes, nhid=256, ntokens_per_class=ntokens_per_class)

#         def forward(self, x, y):
#             x = nn.functional.relu(self.model(x))
#             x = nn.functional.relu(self.linear(x))
#             loss, target_probs, layer_top_probs, layer_bottom_probs, top_indx, botton_indx, real_indx, preds = self.hs(x, y)
#             return loss, real_indx, preds

#     return TIMM(pretrained, pretrained_version, **kwargs)

@register
def htimm_model(pretrained=False, pretrained_version="vit_base_patch16_224", **kwargs):

    class TIMM(nn.Module):

        def __init__(self, pretrained, pretrained_version, num_classes, ntokens_per_class, **kwargs):
            super(TIMM, self).__init__()
            self.model = timm.create_model(pretrained_version, pretrained=pretrained, num_classes=1000)
            self.hs = nn.AdaptiveLogSoftmaxWithLoss(1000, num_classes, cutoffs=[round(num_classes/3),2*round(num_classes/3)], div_value=4)


        def forward(self, x, y):
            x = nn.functional.relu(self.model(x))
            with torch.cuda.amp.autocast(False):
                z = self.hs(x.float(), y)
                return z[1], z[0], self.hs.log_prob(x.float())

    return TIMM(pretrained, pretrained_version, **kwargs)


@register
def vit(pretrained=False, pretrained_version="vit_base_patch16_224", **kwargs):

    class ViT(nn.Module):

        def __init__(self, pretrained, pretrained_version, num_classes, **kwargs):
            super(ViT, self).__init__()
            self.model = timm.create_model(pretrained_version, pretrained=pretrained, num_classes=num_classes)

        def change_head(self, num_classes):
            self.model.head = nn.Linear(
                in_features=self.model.head.in_features,
                out_features=num_classes
            )

        def forward(self, x, y):
            return self.model(x)

    return ViT(pretrained, pretrained_version, **kwargs)

# @register
# def vitdino(pretrained=False, **kwargs):

#     class ViT(nn.Module):

#         def __init__(self, pretrained, num_classes, **kwargs):
#             super(ViT, self).__init__()
#             module = timm.create_model('vit_base_patch16_224_dino', pretrained=pretrained, num_classes=num_classes)
#             self.model = torch.compile(module)

#         def forward(self, x, y):
#             return self.model(x)

#     return ViT(pretrained, **kwargs)

@register
def hvit(pretrained=False, pretrained_version="vit_base_patch16_224", **kwargs):

    class HViTDino(nn.Module):

        def __init__(self, pretrained, pretrained_version, num_classes, ntokens_per_class, **kwargs):
            super(HViTDino, self).__init__()
            self.model = timm.create_model(pretrained_version, pretrained=pretrained, num_classes=256)
            self.hs = HierarchicalSoftmax(ntokens=num_classes, nhid=256, ntokens_per_class=ntokens_per_class)

        def forward(self, x, y):
            x = self.model(x)
            x = nn.functional.relu(x)
            loss, target_probs, layer_top_probs, layer_bottom_probs, top_indx, botton_indx, real_indx, preds = self.hs(x, y)
            return loss, real_indx, preds

    return HViTDino(pretrained, pretrained_version, **kwargs)


