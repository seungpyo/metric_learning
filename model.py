from enum import Enum
import torch
from torch import nn
from torchvision import models
from typing import *

class ModelType(Enum):
    RESNET18 = "resnet18"
    RESNET50 = "resnet50"
    RESNET101 = "resnet101"
    
def get_backbone(model_type: ModelType, pretrained: bool = True) -> nn.Module:
    if model_type == ModelType.RESNET18:
        model = models.resnet18(pretrained=pretrained)
        num_features = model.fc.in_features
        model.fc = nn.Identity()
        return {"model": model, "num_features": num_features}
    elif model_type == ModelType.RESNET50:
        model = models.resnet50(pretrained=pretrained)
        num_features = model.fc.in_features
        model.fc = nn.Identity()
        return {"model": model, "num_features": num_features}
    elif model_type == ModelType.RESNET101:
        model = models.resnet101(pretrained=pretrained)
        num_features = model.fc.in_features
        model.fc = nn.Identity()
        return {"model": model, "num_features": num_features}
    else:
        raise ValueError("Invalid model type")
    
class EmbeddingModel(nn.Module):
    def __init__(self, embedding_dims: Union[int, List[int]]) -> None:
        super().__init__()
        backbone_dict = get_backbone(ModelType.RESNET18)
        self.backbone = backbone_dict["model"]
        self.num_features = backbone_dict["num_features"]
        if isinstance(embedding_dims, int):
            self.fc = nn.Linear(self.num_features, embedding_dims)
        elif isinstance(embedding_dims, list):
            layers = []
            for i in range(len(embedding_dims) - 1):
                layers.append(nn.Linear(embedding_dims[i], embedding_dims[i + 1]))
                layers.append(nn.ReLU())
            self.fc = nn.Sequential(*layers)
        else:
            raise TypeError(f"Invalid type {type(embedding_dims)} for embedding_dims")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.fc(x)
        return x
