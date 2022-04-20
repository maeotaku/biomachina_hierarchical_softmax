import os
import pickle
import warnings

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, StochasticWeightAveraging
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.loggers import WandbLogger
from tqdm import tqdm_notebook as tqdm


from our_datasets import PlatCLEFSimCLR, PlantCLEF2022Supr
from engines import SimCLREngine, SuprEngine
from models.factory import create_model

from summary import *

warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)


CODE_ROOT = '/workspace/biomachina/'

import sys
sys.path.insert(0, CODE_ROOT)

import os
from hydra import initialize, initialize_config_module, initialize_config_dir, compose
from omegaconf import OmegaConf
initialize_config_dir(config_dir=os.path.join(CODE_ROOT, "config"))



def get_engine(cfg, loader, loader_val, model, class_dim, epochs):
    if cfg.engine.name == "simclr":
        return SimCLREngine(model=model, loader=loader, loader_val=loader_val, cfg=cfg, epochs=epochs)
    else:
        return SuprEngine(model=model, loader=loader, loader_val=loader_val, cfg=cfg, epochs=epochs, class_dim=class_dim)


def get_model(cfg, original_path, num_classes):
    model_cfg = OmegaConf.to_container(cfg.model)
    model_cfg["num_classes"] = num_classes
    pretrained = cfg.pretrained_model
    checkpoint_dir = get_full_path(original_path, cfg.model_checkpoints)
    model_check_point = os.path.join(checkpoint_dir, cfg.pretrained_model_point)
    return create_model(model_name=cfg.model.name, pretrained=pretrained, checkpoint_path=model_check_point,
                        **model_cfg)


def get_exp_name(cfg):
    return f"{cfg.name}-ds={cfg.dataset.name}"


def get_full_path(base_path, path):
    r"""
        Expands environment variables and user alias (~ tilde), in the case of relative paths it uses the base path
        to create a full path.

        args:
            base_path: used in case of path is relative path to expand the path.
            path: directory to be expanded. i.e data, ./web, ~/data, $HOME, %USER%, /data
    """
    eval_path = os.path.expanduser(os.path.expandvars(path))
    return eval_path if os.path.isabs(eval_path) else os.path.join(base_path, eval_path)


def get_dataset(original_path, cfg):
    columns = ["classid", "image_path", "species", "genus", "family"]
    ds_root = get_full_path(original_path, cfg.dataset.path)
    df = pd.read_csv(os.path.join(ds_root, cfg.dataset.csv), sep=";", usecols=columns)
    # df = df.head(n=10000)
    if cfg.dataset.name == "web":
        ds = PlatCLEFSimCLR(df, root=os.path.join(ds_root, cfg.dataset.images),
                            label_col=cfg.dataset.label_col,
                            filename_col=cfg.dataset.filename_col,
                            size=cfg.resolution)
        return ds, ds, None
    if cfg.dataset.name == "trusted":
        from os.path import exists
        path = os.path.join(original_path, "trusted_ds_cache.pickle")
        if exists(path):
            ds = pickle.load(open(path, "rb"))
            print("Dataset cache loaded.")
        else:
            transform = transforms.Compose([transforms.Resize([cfg.resolution, cfg.resolution]),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                            ])
            ds = PlantCLEF2022Supr(df, root=os.path.join(ds_root, cfg.dataset.images),
                                   label_col=cfg.dataset.label_col, filename_col=cfg.dataset.filename_col,
                                   transform=transform)
            pickle.dump(ds, open(path, "wb"))
        dst, dsv = ds.split(train_perc=cfg.dataset.train_perc)
        return ds, dst, dsv
    
    
    

# @hydra.main(config_path="config", config_name="simclr_vit.yaml")
# @hydra.main(config_path="config", config_name="supr_hresnet50.yaml")
# @hydra.main(config_path="config", config_name="supr_hresnet101.yaml")
# @hydra.main(config_path="config", config_name="supr_vitae.yaml")
# @hydra.main(config_path="config", config_name="supr_hefficientnet_b4.yaml")
# @hydra.main(config_path="config", config_name="supr_hcct_14_7x2_224.yaml")
# @hydra.main(config_path="config", config_name="supr_hdensenet.yaml")
# @hydra.main(config_path="config", config_name="supr_hefficientnet_b4.yaml")
def execute_training(cfg: DictConfig):
    exp_name = get_exp_name(cfg)
    original_path = CODE_ROOT # hydra.utils.get_original_cwd()
    ds, dst, dsv = get_dataset(original_path=original_path, cfg=cfg)
    loader = torch.utils.data.DataLoader(dst, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
                                         num_workers=cfg.num_workers, persistent_workers=True, pin_memory=True)
    loader_val = torch.utils.data.DataLoader(dsv, batch_size=cfg.batch_size, shuffle=False, drop_last=True,
                                             num_workers= cfg.num_workers, pin_memory=True,
                                             persistent_workers=True) if dsv else None
                                             

    print(ds)
    print(dst)
    print(dsv)
    model = get_model(cfg=cfg, original_path=original_path, num_classes=ds.class_size)
    # summary(model, (3, cfg.resolution, cfg.resolution), (), device='cpu')
    engine = get_engine(cfg=cfg, loader=loader, loader_val=loader_val, model=model, class_dim=ds.class_size, epochs=cfg.epochs)

    from pytorch_lightning.callbacks import ModelCheckpoint

    engine_checkpoint_callback = ModelCheckpoint(
        monitor="train_loss",
        dirpath=os.path.join(original_path, cfg.engine_checkpoints),
        save_top_k=1,
        mode="min",
        filename=f"{exp_name}" + "-{epoch}-{train_loss:.2f}-{train_acc:.2f}--{val_loss:.2f}-{val_acc:.2f}",
        save_on_train_epoch_end=True
    )

    # model_checkpoint_callback = ModelCheckpoint(
    #     monitor="train_loss",
    #     dirpath=os.path.join(original_path, cfg.model_checkpoints),
    #     save_top_k=1,
    #     mode="min",
    #     filename=f"{exp_name}" + "-{epoch}-{train_loss:.2f}-{train_acc:.2f}--{val_loss:.2f}-{val_acc:.2f}",
    #     save_on_train_epoch_end=True,
    #     every_n_train_steps=int((len(ds) / cfg.batch_size) * cfg.time_per_epoch),
    #     save_weights_only=True
    # )

    #tb_logger = pl_loggers.TensorBoardLogger(name=exp_name, save_dir=os.path.join(original_path, "logs"))
    wandb_logger = WandbLogger(name=exp_name, project="plantclef2022", entity="tesiarioscarranza")

    callbacks = [ engine_checkpoint_callback, StochasticWeightAveraging(swa_lrs=1e-2)]
    val_loaders = []
    if dsv:
        callbacks.append(EarlyStopping(monitor="val_loss"))
        val_loaders.append(loader_val)

#     trainer = pl.Trainer(accelerator="tpu", tpu_cores=8, max_epochs=cfg.epochs, progress_bar_refresh_rate=20,
#                          callbacks=callbacks, logger=[ tb_logger ], limit_train_batches=1.0, gradient_clip_val=0.5, precision=16, num_sanity_val_steps=0)
#                         #  ,
#                         #  gradient_clip_val=0.5, num_sanity_val_steps=0)
    trainer = pl.Trainer(gpus=2, precision=cfg.precision, max_epochs=cfg.epochs, strategy='ddp',
                         callbacks=callbacks, logger=[wandb_logger], limit_train_batches=1.0, amp_backend="native",
                         gradient_clip_val=0.5, accumulate_grad_batches=32, num_sanity_val_steps=0)

    if cfg.last_engine_checkpoint:
        trainer.fit(engine, train_dataloaders=loader, val_dataloaders=val_loaders,
                    ckpt_path=os.path.join(original_path, cfg.engine_checkpoints, cfg.last_engine_checkpoint))
    else:
        trainer.fit(engine, train_dataloaders=loader, val_dataloaders=val_loaders)





cfg=compose(config_name="supr_hefficientnet_b4")
print(cfg)



execute_training(cfg)