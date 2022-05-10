import torch
import torchvision.models as models
from torch import nn
# from torchvision.models import EfficientNet

from .registry import register
from .hsoftmax import HierarchicalSoftmax

class ObservationTransformer(nn.Module):

    def __init__(self, encoder: nn.Module, nhead=10, feature_dim=1000, output_size=128):
        super().__init__()
        self.feature_dim = feature_dim
        self.encoder = encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=nhead, batch_first=True)
        # self.decoder = nn.Linear(feature_dim * 2, output_size)
        self.transformer = nn.TransformerEncoder(encoder_layer=encoder_layers, num_layers=3)
        self.decoder = nn.Linear(feature_dim, output_size)

    def forward(self, x):
        B, S, C, R, _ = x.shape
        x = x.reshape(-1, C, R, R)  # squeeze observations to (batch * sequence, channel, width, height)
        # size into a single dim for encoding as batch
        features = self.encoder(x)  # encode all images for B observations of S images each
        features = features.reshape(B, S, self.feature_dim)  # resize back to observation sizes (batch, sequence, )
        # out = self.head(features)  # apply attention to teh features
        # return self.decoder(out.reshape(B, S * self.feature_dim))
        output = self.transformer(features)
        output = self.decoder(output)
        return output.mean(dim=1)


class HObservationTransformer(nn.Module):

    def __init__(self, encoder: nn.Module, nhead=10, feature_dim=1000, num_classes=80000, ntokens_per_class=80, **kwargs):
        super().__init__()
        self.obs_transformer = ObservationTransformer(encoder=encoder, nhead=nhead, feature_dim=feature_dim,
                                                      output_size=128)
        self.hs = HierarchicalSoftmax(ntokens=num_classes, nhid=128, ntokens_per_class=ntokens_per_class)

    def forward(self, x, y=None):
        assert len(x.shape) == 5, "Observation shape should be (batch, sequence, channels, witdh, height)"
        x = self.obs_transformer(x)
        if y is not None:
            loss, target_probs, layer_top_probs, layer_bottom_probs, top_indx, botton_indx, real_indx, preds = self.hs(x, y)
            return loss, real_indx, preds
        else:
            preds = self.hs(x)
            return preds


