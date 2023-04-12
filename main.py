import os
import pickle
import warnings
from datetime import timedelta

import albumentations as A
import hydra
import pandas as pd
import pytorch_lightning as pl
import torch
import wandb
from engines import SuprEngine, CombinedSuprEngine #SimCLREngine
from models.factory import create_model
from omegaconf import DictConfig, OmegaConf
from our_datasets import (  # PlantCLEF2022Supr #PlatCLEFSimCLR, ObservationsDataset
    get_dataset, get_full_path)
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         StochasticWeightAveraging)
from pytorch_lightning.loggers import WandbLogger
from summary import *

warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

torch.set_float32_matmul_precision('high')

CODE_ROOT = f'/home/maeotaku/Documents/code/biomachina/'

import sys

sys.path.insert(0, CODE_ROOT)

import os

from hydra import (compose, initialize, initialize_config_dir,
                   initialize_config_module)
from omegaconf import OmegaConf


def get_engine(cfg, name, model, datamodule, class_dim, epochs):
    # if name == "simclr":
        # return SimCLREngine(model=model, loader=loader, loader_val=loader_val, cfg=cfg, epochs=epochs)
    # else:
    # return CombinedSuprEngine(model=model, datamodule=datamodule, cfg=cfg, epochs=epochs)
    return SuprEngine(model=model, cfg=cfg, epochs=epochs, class_dim=class_dim)


def get_model(cfg, original_path, num_classes):
    model_cfg = OmegaConf.to_container(cfg.model)
    model_cfg["num_classes"] = num_classes
    pretrained = cfg.pretrained_model
    checkpoint_dir = get_full_path(original_path, cfg.model_checkpoints)
    model_check_point = os.path.join(checkpoint_dir, cfg.pretrained_model_point)
    return create_model(model_name=cfg.model.name, pretrained=pretrained, checkpoint_path=model_check_point, pretrained_version=cfg.pretrained_version,
                        **model_cfg)


def get_exp_name(cfg):
    if "pretrained_version" in cfg:
        return f"{cfg.pretrained_version}"
    return f"{cfg.name}-ds={cfg.dataset.name}"




@hydra.main(config_path="config", config_name="supr_timm.yaml")
# @hydra.main(config_path="config", config_name="supr_htimm.yaml")
def execute_training(cfg: DictConfig):
    exp_name = get_exp_name(cfg)
    original_path = CODE_ROOT
   
    datamodule = get_dataset(original_path, cfg=cfg)
    datamodule.setup()

    model = get_model(cfg=cfg, original_path=original_path, num_classes=datamodule.class_size)

    engine = get_engine(name=cfg.engine.name, cfg=cfg, model=model, datamodule=datamodule, class_dim=datamodule.class_size, epochs=cfg.epochs)

    from pytorch_lightning.callbacks import ModelCheckpoint

    if cfg.wandb_id:
        wandb_logger = WandbLogger(name=exp_name, project=cfg.wandb_project, entity=cfg.wandb_entity, id=cfg.wandb_id, resume="allow")
    else:
        wandb_logger = WandbLogger(name=exp_name, project=cfg.wandb_project, entity=cfg.wandb_entity)
    wandb.init(config=cfg.__dict__, allow_val_change=True)
    wandb.config.update(cfg, allow_val_change=True)
    
    engine_checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        dirpath=os.path.join(original_path, cfg.engine_checkpoints),
        save_top_k=1,
        mode="max",
        filename=f"{exp_name}-{wandb.run.id}" + "-{epoch}-{train_loss:.2f}-{train_acc:.2f}--{val_loss:.2f}-{val_acc:.2f}",
        train_time_interval=timedelta(minutes=15)
    )

    callbacks = [ engine_checkpoint_callback, LearningRateMonitor(logging_interval='step')] #StochasticWeightAveraging(swa_lrs=1e-2),
    callbacks.append(EarlyStopping(monitor="val_loss"))

    trainer = pl.Trainer(precision=cfg.precision, max_epochs=cfg.epochs, accelerator="gpu", devices=2, strategy="ddp", num_nodes=1,
                        # accumulate_grad_batches=10,
                        callbacks=callbacks, logger=[ wandb_logger ], limit_train_batches=1.0)
                        # gradient_clip_val=0.25) #,  num_sanity_val_steps=0)

    if not cfg.last_engine_checkpoint.path:
        trainer.fit(engine, datamodule=datamodule)
    else:
        prev_engine_path = os.path.join(original_path, cfg.engine_checkpoints, cfg.last_engine_checkpoint.path)
        # if cfg.engine == cfg.last_engine_checkpoint.engine:
        print("Training and loading.")
        trainer.fit(engine, datamodule=datamodule, ckpt_path=prev_engine_path)
        # else: 
        #     prev_engine = get_engine(name=cfg.last_engine_checkpoint.engine, cfg=cfg, ds=dst, loader=loader, loader_val=loader_val, model=model, class_dim=512, epochs=cfg.epochs)
        #     prev_engine.load_from_checkpoint(prev_engine_path, cfg=cfg, ds=dst, loader=loader, loader_val=loader_val, model=model, epochs=cfg.epochs)
        #     # extract model from previous engine into new one
        #     engine.model = prev_engine.model

        #     # engine.model.change_head(num_classes=ds.class_size)
        #     trainer.fit(engine, train_dataloaders=loader, val_dataloaders=val_loaders)


if __name__ == "__main__":
    execute_training()
