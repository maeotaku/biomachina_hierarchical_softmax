import os

import hydra
import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets import PlatCLEFSimCLR, PlantCLEF2022Supr
from engines import SimCLREngine, SuprEngine
from models.factory import create_model


def get_engine(cfg, loader, loader_val, model, class_dim):
    if cfg.engine.name == "simclr":
        return SimCLREngine(model=model, loader=loader, loader_val=loader_val, cfg=cfg)
    else:
        return SuprEngine(model=model, loader=loader, loader_val=loader_val, cfg=cfg, class_dim=class_dim)


def get_model(cfg, original_path, num_classes):
    model_cfg = OmegaConf.to_container(cfg.model)
    model_cfg["num_classes"] = num_classes
    checkpoint_dir = get_full_path(original_path, cfg.model_checkpoints)
    model_check_point = os.path.join(checkpoint_dir, cfg.pretrained_model_point)
    return create_model(model_name=cfg.model.name, checkpoint_path=model_check_point, **model_cfg)


def get_exp_name(cfg):
    return f"{cfg.name}-engine={cfg.engine.name}-dataset={cfg.dataset.name}-model={cfg.model.name}"


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
        transform = transforms.Compose([transforms.Resize([cfg.resolution, cfg.resolution]),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                        ])
        ds = PlantCLEF2022Supr(df, root=os.path.join(ds_root, cfg.dataset.images),
                               label_col=cfg.dataset.label_col, filename_col=cfg.dataset.filename_col,
                               transform=transform)
        dst, dsv = ds.split(train_perc=cfg.dataset.train_perc)
        return ds, dst, dsv


# @hydra.main(config_path="config", config_name="simclr_vit.yaml")
@hydra.main(config_path="config", config_name="supr_resnet.yaml")
def run(cfg: DictConfig):
    exp_name = get_exp_name(cfg)
    original_path = hydra.utils.get_original_cwd()
    ds, dst, dsv = get_dataset(original_path=original_path, cfg=cfg)
    loader = torch.utils.data.DataLoader(dst, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
                                         num_workers=cfg.num_workers, persistent_workers=True)
    loader_val = torch.utils.data.DataLoader(dsv, batch_size=cfg.batch_size, shuffle=False, drop_last=True,
                                             num_workers=cfg.num_workers, persistent_workers=True) if dsv else None

    print(ds)
    print(dst)
    print(dsv)
    model = get_model(cfg=cfg, original_path=original_path, num_classes=ds.class_size)
    engine = get_engine(cfg=cfg, loader=loader, loader_val=loader_val, model=model, class_dim=ds.class_size)

    from pytorch_lightning.callbacks import ModelCheckpoint

    engine_checkpoint_callback = ModelCheckpoint(
        monitor="train_loss",
        dirpath=os.path.join(original_path, cfg.engine_checkpoints),
        save_top_k=2,
        mode="min",
        filename=f"{exp_name}" + "-{epoch}-{train_loss:.2f}-{train_acc:.2f}--{val_loss:.2f}-{val_acc:.2f}",
        save_on_train_epoch_end=True,
        every_n_train_steps=int((len(ds) / cfg.batch_size) * cfg.time_per_epoch)
    )

    model_checkpoint_callback = ModelCheckpoint(
        monitor="train_loss",
        dirpath=os.path.join(original_path, cfg.model_checkpoints),
        save_top_k=2,
        mode="min",
        filename=f"{exp_name}" + "-{epoch}-{train_loss:.2f}-{train_acc:.2f}--{val_loss:.2f}-{val_acc:.2f}",
        save_on_train_epoch_end=True,
        every_n_train_steps=int((len(ds) / cfg.batch_size) * cfg.time_per_epoch),
        save_weights_only=True
    )

    tb_logger = pl_loggers.TensorBoardLogger(name=exp_name, save_dir=os.path.join(original_path, "logs/"))
    wandb_logger = WandbLogger()

    callbacks = [model_checkpoint_callback, engine_checkpoint_callback]
    val_loaders = []
    if dsv:
        callbacks.append(EarlyStopping(monitor="val_loss"))
        val_loaders.append(loader_val)

    trainer = pl.Trainer(gpus=1, num_nodes=1, precision=cfg.precision,
                         callbacks=callbacks, logger=[tb_logger])
    # , accumulate_grad_batches=cfg.accumulate_batches)
    # trainer.tune(engine)

    if cfg.last_engine_checkpoint:
        trainer.fit(engine, train_dataloader=loader, val_dataloaders=val_loaders,
                    ckpt_path=os.path.join(original_path, cfg.engine_checkpoints, cfg.last_engine_checkpoint))
    else:
        trainer.fit(engine, train_dataloader=loader, val_dataloaders=val_loaders)


if __name__ == "__main__":
    run()
