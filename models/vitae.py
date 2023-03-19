from models import register
from .vitae_module.vitmodules import ViTAE_ViT_basic
from .hsoftmax import HierarchicalSoftmax
import torch.nn as nn
from  torch.cuda.amp import autocast


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225),
        'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'ViTAE_basic_7': _cfg(),
}


#
# #  The tiny model
# @register_model
# def ViTAE_basic_7(pretrained=False, **kwargs): # adopt performer for tokens to token
#     model = ViTAE_ViT_basic(RC_tokens_type=['performer', 'performer', 'performer_less'], NC_tokens_type=['transformer', 'transformer', 'transformer'], stages=3, embed_dims=[64, 64, 128], token_dims=[64, 64, 256],
#                             downsample_ratios=[4, 2, 2], NC_depth=[0, 0, 7], NC_heads=[1, 1, 4], RC_heads=[1, 1, 1], mlp_ratio=2., NC_group=[1, 1, 64], RC_group=[1, 1, 1], **kwargs)
#     model.default_cfg = default_cfgs['ViTAE_basic_7']
#     if pretrained:
#         load_pretrained(
#             model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
#     return model
#
# # The small model
# @register_model
# def ViTAE_basic_14(pretrained=False, **kwargs): # adopt performer for tokens to token
#     model = ViTAE_ViT_basic(RC_tokens_type=['performer', 'performer', 'performer_less'], NC_tokens_type=['transformer', 'transformer', 'transformer'], stages=3, embed_dims=[64, 64, 192], token_dims=[64, 64, 384],
#                             downsample_ratios=[4, 2, 2], NC_depth=[0, 0, 14], NC_heads=[1, 1, 6], RC_heads=[1, 1, 1], mlp_ratio=3., NC_group=[1, 1, 96], RC_group=[1, 1, 1], **kwargs)
#     model.default_cfg = default_cfgs['ViTAE_basic_7']
#     if pretrained:
#         load_pretrained(
#             model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
#     return model

# The 6M model
def ViTAE_basic_10(pretrained=False, **kwargs):  # adopt performer for tokens to token
    model = ViTAE_ViT_basic(RC_tokens_type=['performer', 'performer', 'performer_less'],
                            NC_tokens_type=['transformer', 'transformer', 'transformer'], stages=3,
                            embed_dims=[64, 64, 128], token_dims=[64, 64, 256],
                            downsample_ratios=[4, 2, 2], NC_depth=[0, 0, 10], NC_heads=[1, 1, 4], RC_heads=[1, 1, 1],
                            mlp_ratio=2., NC_group=[1, 1, 64], RC_group=[1, 1, 1], **kwargs)
    model.default_cfg = default_cfgs['ViTAE_basic_7']
    # if pretrained:
    #     load_pretrained(
    #         model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

def ViTAE_basic_11(pretrained=False, **kwargs): # adopt performer for tokens to token
    model = ViTAE_ViT_basic(RC_tokens_type=['performer', 'performer', 'performer_less'], NC_tokens_type=['transformer', 'transformer', 'transformer'], stages=3, embed_dims=[64, 64, 160], token_dims=[64, 64, 320],
                            downsample_ratios=[4, 2, 2], NC_depth=[0, 0, 11], NC_heads=[1, 1, 5], RC_heads=[1, 1, 1], mlp_ratio=2., NC_group=[1, 1, 64], RC_group=[1, 1, 1], **kwargs)
    model.default_cfg = default_cfgs['ViTAE_basic_7']
    # if pretrained:
    #     load_pretrained(
    #         model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

class HierarchicalVITAE(nn.Module):

    def __init__(self, pretrained, num_classes, ntokens_per_class, **kwargs):
        super(HierarchicalVITAE, self).__init__()
        import timm
        self.backbone = timm.create_model('vit_base_patch16_224', pretrained=pretrained)

        # self.backbone = ViTAE_basic_11(pretrained, **kwargs)

        # add mlp projection head
        # dim_mlp = self.backbone.head.in_features
        # self.backbone.head = nn.Linear(dim_mlp, num_classes)

        # self.backbone.head = nn.Linear(dim_mlp, dim_mlp)
        # dim_mlp = self.backbone.fc.out_features
        # self.fc = nn.Linear(dim_mlp, num_classes)
        self.hs = HierarchicalSoftmax(ntokens=num_classes, nhid=1000, ntokens_per_class=ntokens_per_class)
    #
    # def freeze(self):
    #     for param in self.backbone.layer1.parameters():
    #         param.requires_grad = False
    #     for param in self.backbone.layer2.parameters():
    #         param.requires_grad = False

    def forward(self, x, y):
        # return self.backbone(x)
        x = self.backbone(x)
        x = nn.functional.relu(x)
        loss, target_probs, layer_top_probs, layer_bottom_probs, top_indx, botton_indx, real_indx, preds = self.hs(x, y)
        return loss, real_indx, preds


#
# # The 13M model
# @register_model
# def ViTAE_basic_11(pretrained=False, **kwargs): # adopt performer for tokens to token
#     model = ViTAE_ViT_basic(RC_tokens_type=['performer', 'performer', 'performer_less'], NC_tokens_type=['transformer', 'transformer', 'transformer'], stages=3, embed_dims=[64, 64, 160], token_dims=[64, 64, 320],
#                             downsample_ratios=[4, 2, 2], NC_depth=[0, 0, 11], NC_heads=[1, 1, 5], RC_heads=[1, 1, 1], mlp_ratio=2., NC_group=[1, 1, 64], RC_group=[1, 1, 1], **kwargs)
#     model.default_cfg = default_cfgs['ViTAE_basic_7']
#     if pretrained:
#         load_pretrained(
#             model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
#     return model

@register
def vitae10(pretrained=False, **kwargs):
    #return ViTAE_basic_10(pretrained, **kwargs)
    return HierarchicalVITAE(pretrained, **kwargs)
