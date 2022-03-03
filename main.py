import hydra
from omegaconf import DictConfig

import os

import hydra
import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets import PlatCLEFSimCLR, PlantCLEF2022Supr
from engines import SimCLREngine, SuprEngine
from models import ResNetSelfSupr, ResNetClassifier


def get_engine(cfg, loader, loader_val, model):
    engines = {"simclr": SimCLREngine, "supr": SuprEngine}
    return engines[cfg.engine.name](model=model, loader=loader, loader_val=loader_val, cfg=cfg)


def get_model(cfg, original_path, class_size):
    if cfg.model.name == "resnet":
        if cfg.engine.name == "supr":
            saved_model = ResNetSelfSupr(base_model=cfg.model.arch, out_dim=cfg.model.out_dim)
            _ = SimCLREngine.load_from_checkpoint(os.path.join(original_path, cfg.chekpoint_point),
                                                  model=saved_model, loader=None, loader_val=None, cfg=cfg)
            return ResNetClassifier(saved_model, feature_dim=cfg.model.out_dim, class_dim=class_size)
        else:
            return ResNetSelfSupr(base_model=cfg.model.arch, out_dim=cfg.model.out_dim)


def get_dataset(original_path, cfg):
    columns = ["classid", "image_path", "species", "genus", "family"]
    df = pd.read_csv(os.path.join(original_path, cfg.dataset.csv), sep=";", usecols=columns)
    if cfg.dataset.name == "web":
        ds = PlatCLEFSimCLR(df, root=os.path.join(original_path, cfg.dataset.image_root),
                              label_col=cfg.dataset.label_col,
                              filename_col=cfg.dataset.filename_col,
                              size=cfg.resolution)
        return ds, ds, None
    if cfg.dataset.name == "trusted":
        transform = transforms.Compose([transforms.Resize([cfg.resolution, cfg.resolution]),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                        ])
        ds = PlantCLEF2022Supr(df, root=os.path.join(original_path, cfg.dataset.image_root),
                                 label_col=cfg.dataset.label_col, filename_col=cfg.dataset.filename_col,
                                 transform=transform)
        dst, dsv = ds.split(train_perc=cfg.dataset.train_perc)
        return ds, dst, dsv


@hydra.main(config_path="config", config_name="simclr.yaml")
# @hydra.main(config_path="config", config_name="supr.yaml")
def run(cfg: DictConfig):
    original_path = hydra.utils.get_original_cwd()
    ds, dst, dsv = get_dataset(original_path=original_path, cfg=cfg)
    loader = torch.utils.data.DataLoader(dst, batch_size=cfg.dataset.batch_size, shuffle=True, drop_last=True,
                                         num_workers=cfg.num_workers, persistent_workers=True)
    loader_val = torch.utils.data.DataLoader(dst, batch_size=cfg.dataset.batch_size, shuffle=True, drop_last=True,
                                             num_workers=cfg.num_workers, persistent_workers=True) if dsv else None

    print(loader)
    print(loader_val)
    model = get_model(cfg=cfg, original_path=original_path, class_size=ds.class_size)
    engine = get_engine(cfg=cfg, loader=loader, loader_val=loader_val, model=model)

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

    callbacks = [checkpoint_callback]
    val_loaders = []
    if dsv:
        callbacks.append(EarlyStopping(monitor="val_loss"))
        val_loaders.append(loader_val)
    trainer = pl.Trainer(gpus=1, num_nodes=1, precision=cfg.precision,
                         callbacks=callbacks, logger=[tb_logger])
    trainer.fit(engine, train_dataloader=loader, val_dataloaders=val_loaders)


if __name__ == "__main__":
    run()
