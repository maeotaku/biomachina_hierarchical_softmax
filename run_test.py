import os
import pickle
import warnings
from datetime import timedelta

import albumentations as A
import hydra
import pandas as pd
import pytorch_lightning as pl
import torch
from engines import TestEngine, SuprEngine
from models.factory import create_model
from omegaconf import DictConfig, OmegaConf
from our_datasets import PlantTestOnlyDataModule, PlantDataModule, get_full_path
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         StochasticWeightAveraging)
from pytorch_lightning.loggers import WandbLogger

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


def get_model(cfg, original_path, num_classes):
    model_cfg = OmegaConf.to_container(cfg.model)
    model_cfg["num_classes"] = num_classes
    pretrained = cfg.pretrained_model
    checkpoint_dir = get_full_path(original_path, cfg.model_checkpoints)
    model_check_point = os.path.join(checkpoint_dir, cfg.pretrained_model_point)
    return create_model(model_name=cfg.model.name, pretrained=pretrained, checkpoint_path=model_check_point, pretrained_version=cfg.pretrained_version,
                        **model_cfg)


@hydra.main(config_path="config", config_name="test_supr_timm.yaml")
def execute_test(cfg: DictConfig):
    original_path = CODE_ROOT

    origina_data =  PlantDataModule(original_path, cfg=cfg)
    origina_data.setup()
   
    datamodule = PlantTestOnlyDataModule(original_path, cfg=cfg, class_dict=origina_data.class_dict, inv_class_dict=origina_data.inv_class_dict)
    datamodule.setup()





    trainer = pl.Trainer(
        precision=cfg.precision, max_epochs=cfg.epochs, accelerator="gpu", devices=2, strategy="ddp", num_nodes=1
    )

    model = get_model(cfg=cfg, original_path=original_path, num_classes=datamodule.class_size)

    prev_engine_path = os.path.join(original_path, cfg.engine_checkpoints, cfg.last_engine_checkpoint.path)
    prev_engine = SuprEngine(cfg=cfg, model=model, class_dim=80000, epochs=1)
    prev_engine.load_from_checkpoint(prev_engine_path, cfg=cfg, model=model, epochs=1,  class_dim=80000)

    test_engine = TestEngine(CODE_ROOT, cfg=cfg, model=prev_engine.model, datamodule=datamodule, class_dict=origina_data.class_dict, inv_class_dict=origina_data.inv_class_dict)

    trainer.test(test_engine, datamodule=datamodule)
    

if __name__ == "__main__":
    execute_test()
