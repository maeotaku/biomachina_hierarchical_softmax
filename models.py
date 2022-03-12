import torch.nn as nn
import torchvision.models as models

class ResNetSelfSupr(nn.Module):

    def __init__(self, base_model, out_dim):
        super(ResNetSelfSupr, self).__init__()
        self.backbone = models.resnet50(pretrained=False, num_classes=out_dim)
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

    def __init__(self, model : ResNetSelfSupr, feature_dim, class_dim : int):
        super(ResNetClassifier, self).__init__()
        self.model = model
        self.fc = nn.Linear(feature_dim, class_dim)
        
    def forward(self, x):
        x = self.model(x)
        return self.fc(x)
        
        
    
 
    