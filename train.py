import pytorch_lightning as pl
from model import ModelType, EmbeddingModel
import torch
from torch.optim import AdamW
from typing import *
from dataclasses import dataclass
from cars196 import get_dataset
from PIL import Image
import numpy as np
import albumentations as A
from torchvision.datasets import StanfordCars

@dataclass
class LiftedStructureEmbeddingConfig:
    model_type: ModelType
    embedding_dim: Union[int, List[int]]
    lr: float
    weight_decay: float
    max_epochs: int
    batch_size: int

class LiftedStructureEmbedding(pl.LightningModule):
    def __init__(self, config: LiftedStructureEmbeddingConfig) -> None:
        super().__init__()
        self.config = config
        self.model = EmbeddingModel(config.embedding_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def training_step(self, batch, batch_idx) -> Dict[str, Any]:
        return batch.sum()
        
    
    def validation_step(self, batch, batch_idx) -> Dict[str, Any]:
        return batch.sum()
    
    def configure_optimizers(self) -> Dict[str, Any]:
        param_groups = [
            {'params': self.model.backbone.parameters(), 'lr': self.config.lr},
            {'params': self.model.fc.parameters(), 'lr': self.config.lr * 10},
        ]
        optimizer = AdamW(
            params=param_groups,
            lr=self.config.lr, 
            weight_decay=self.config.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.max_epochs)
        return {
            "optimizer": optimizer,
            "scheduler": scheduler,
        }
        
def albumentations_fn(transform: callable, img: Image.Image):
    return transform(image=np.array(img))["image"]

def main():
    '''
    
    model_type: ModelType
    embedding_dim: Union[int, List[int]]
    lr: float
    weight_decay: float
    max_epochs: int
    '''
    config = LiftedStructureEmbeddingConfig(
        model_type=ModelType.RESNET18,
        embedding_dim=64,
        lr=1e-3,
        weight_decay=1e-4,
        max_epochs=100,
        batch_size=32,
    )
        
    module = LiftedStructureEmbedding(config=config)
    tb_logger = pl.loggers.TensorBoardLogger("logs")
    tb_logger.log_hyperparams(config.__dict__)
    
    trainer = pl.Trainer(
        logger=tb_logger,
        log_every_n_steps=1,
        max_epochs=config.max_epochs,
        accelerator="gpu",
    )
    train_t = A.Compose([
        A.Resize(256, 256),
        A.RandomCrop(227, 227),
        A.HorizontalFlip(p=0.5),
    ])
    test_t = A.Compose([
        A.Resize(256, 256),
        A.CenterCrop(227, 227),
    ])
    train_ds = StanfordCars(root="/data/", split="train", download=True, transform=lambda x: albumentations_fn(train_t, x))
    test_ds = StanfordCars(root="/data/", split="test", download=True, transform=lambda x: albumentations_fn(test_t, x))
    
    train_dl = torch.utils.data.DataLoader(
        dataset=train_ds, 
        batch_size=config.batch_size, 
        num_workers=4
    )
    test_dl = torch.utils.data.DataLoader(
        dataset=test_ds,
        batch_size=config.batch_size,
        num_workers=4
    )
    trainer.fit(module, train_dataloaders=train_dl, valid_dataloaders=test_dl)

if __name__ == "__main__":
    main()