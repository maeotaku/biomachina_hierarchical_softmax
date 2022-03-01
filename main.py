import torch
import hydra
from omegaconf import DictConfig

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.loggers import WandbLogger
from models import ResNetSelfSupr, ResNetClassifier
from engines import SimCLREngine, SuprEngine
#
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)
# pd.set_option('max_colwidth', -1)

from datasets import PlatCLEFSimCLR, PlantCLEF2022Supr


def get_engine(cfg, loader, model):
    engines = {"simclr": SimCLREngine, "supr": SuprEngine}
    return engines[cfg.engine.name](model=model, loader=loader, cfg=cfg)


def get_model(cfg, original_path, class_size):
    if cfg.model.name == "resnet":
        if cfg.engine.name == "supr":
            saved_model = ResNetSelfSupr(base_model=cfg.model.arch, out_dim=cfg.model.out_dim)
            _ = SimCLREngine.load_from_checkpoint(os.path.join(original_path, cfg.engine.chekpoint_point),
                                                  model=saved_model, loader=None, cfg=cfg)
            return ResNetClassifier(saved_model, feature_dim=cfg.model.out_dim, class_dim=class_size)
        else:
            return ResNetSelfSupr(base_model=cfg.model.arch, out_dim=cfg.model.out_dim)


def get_dataset(original_path, cfg):
    columns = ["classid", "image_path", "species", "genus", "family"]
    df = pd.read_csv(os.path.join(original_path, cfg.dataset.csv), sep=";", usecols=columns)
    if cfg.dataset.name == "web":
        return PlatCLEFSimCLR(df, root=os.path.join(original_path, cfg.dataset.image_root),
                              label_col=cfg.dataset.label_col,
                              filename_col=cfg.dataset.filename_col,
                              size=cfg.resolution)
    if cfg.dataset.name == "trusted":
        transform = transforms.Compose([transforms.Resize([cfg.resolution, cfg.resolution]),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                        ])
        return PlantCLEF2022Supr(df, root=os.path.join(original_path, cfg.dataset.image_root),
                                 label_col=cfg.dataset.label_col, filename_col=cfg.dataset.filename_col,
                                 transform=transform)


@hydra.main(config_path="config", config_name="supr.yaml")
def run(cfg: DictConfig):
    original_path = hydra.utils.get_original_cwd()
    ds = get_dataset(original_path=original_path, cfg=cfg)
    loader = torch.utils.data.DataLoader(ds, batch_size=cfg.dataset.batch_size, shuffle=True, drop_last=True,
                                         num_workers=cfg.num_workers, persistent_workers=True)

    model = get_model(cfg=cfg, original_path=original_path, class_size=ds.class_size)
    engine = get_engine(cfg=cfg, loader=loader, model=model)

    from pytorch_lightning.callbacks import ModelCheckpoint

    checkpoint_callback = ModelCheckpoint(
        monitor="train_loss",
        dirpath=os.path.join(original_path, cfg.checkpoints),
        save_top_k=2,
        mode="min",
        filename=f"{cfg.engine.name}-" + "{epoch}-{train_loss:.2f}-{train_acc_top1:.2f}",
        save_on_train_epoch_end=True,
        every_n_train_steps=int((len(ds) / cfg.dataset.batch_size) * cfg.time_per_epoch)
    )

    tb_logger = pl_loggers.TensorBoardLogger(os.path.join(original_path, "logs/"))
    wandb_logger = WandbLogger()

    trainer = pl.Trainer(gpus=1, num_nodes=1, precision=cfg.precision, limit_train_batches=0.5,
                         callbacks=[checkpoint_callback], logger=[tb_logger])
    trainer.fit(engine, loader)


if __name__ == "__main__":
    run()
