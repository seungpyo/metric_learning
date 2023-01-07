import pytorch_lightning as pl
from .model import ModelType, EmbeddingModel
import torch
from torch.optim import AdamW
from typing import *
from dataclasses import dataclass
from cars196 import get_dataset


@dataclass
class LiftedStructureEmbeddingConfig:
    model_type: ModelType
    embedding_dim: Union[int, List[int]]
    lr: float
    weight_decay: float
    max_epochs: int

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
    )
        
    module = LiftedStructureEmbedding(config=config)
    
    trainer = pl.Trainer(
        logger=pl.loggers.TensorBoardLogger("logs"),
        log_every_n_steps=1,
        max_epochs=config.max_epochs,
        accelerator="gpu",
    )
    ds = get_dataset("/data/stanford_cars")
    train_dl = torch.utils.data.DataLoader(
        dataset=ds, 
        batch_size=32, 
        num_workers=4
    )
    test_dl = torch.utils.data.DataLoader(
        dataset=ds,
        batch_size=32,
        num_workers=4
    )
    trainer.fit(module, train_dataloaders=train_dl, valid_dataloaders=test_dl)

if __name__ == "__main__":
    main()