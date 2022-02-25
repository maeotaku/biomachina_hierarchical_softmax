import torch.nn as nn
import torchvision.models as models

import pytorch_lightning as pl


class ResNetSelfSupr(pl.LightningModule):

    def __init__(self, base_model, out_dim):
        super(ResNetSelfSupr, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False, num_classes=out_dim),
                            "resnet50": models.resnet50(pretrained=False, num_classes=out_dim)}

        self.backbone = self._get_basemodel(base_model)
        dim_mlp = self.backbone.fc.in_features

        # add mlp projection head
        self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)

    def _get_basemodel(self, model_name):
        model = self.resnet_dict[model_name]
        return model

    def forward(self, x):
        return self.backbone(x)
    
    
class ResNetClassifier(pl.LightningModule):
        
    def __init__(self, model : ResNetSelfSupr, feature_dim, class_dim : int):
        super(ResNetClassifier, self).__init__()
        self.model = model
        self.fc = nn.Linear(feature_dim, class_dim)
        
    def forward(self, x):
        x = self.model(x)
        return self.fc(x)
        
        
    
 
    