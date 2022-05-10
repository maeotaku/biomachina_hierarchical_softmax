import os
import pickle
import warnings

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, StochasticWeightAveraging, LearningRateMonitor
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as pl
import albumentations as A

from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.loggers import WandbLogger
import wandb
from tqdm import tqdm_notebook as tqdm


from our_datasets import PlatCLEFSimCLR, PlantCLEF2022Supr, ObservationsDataset
from engines import SimCLREngine, SuprEngine
from models.factory import create_model

from summary import *

warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)


CODE_ROOT = f'C:/Users/maeot/Documents/code/biomachina'

import sys
sys.path.insert(0, CODE_ROOT)

import os
from hydra import initialize, initialize_config_module, initialize_config_dir, compose
from omegaconf import OmegaConf
# initialize_config_dir(config_dir=os.path.join(CODE_ROOT, "config"))



def get_engine(cfg, ds, loader, loader_val, model, class_dim, epochs):
    if cfg.engine.name == "simclr":
        return SimCLREngine(model=model, loader=loader, loader_val=loader_val, cfg=cfg, epochs=epochs)
    else:
        return SuprEngine(model=model, ds=ds, loader=loader, loader_val=loader_val, cfg=cfg, epochs=epochs, class_dim=class_dim)


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
    columns = ["classid", "image_path", "species", "genus", "family", "gbif_occurrence_id"]
    ds_root = get_full_path(original_path, cfg.dataset.path)
    df = pd.read_csv(os.path.join(ds_root, cfg.dataset.csv), sep=";", usecols=columns)
    # df = df.head(n=1000)
    if cfg.dataset.name == "web":
        ds = PlatCLEFSimCLR(df, root=os.path.join(ds_root, cfg.dataset.images),
                            label_col=cfg.dataset.label_col,
                            filename_col=cfg.dataset.filename_col,
                            size=cfg.resolution)
        return ds, ds, None
    if cfg.dataset.name == "trusted_obs":
        transform = transforms.Compose([transforms.Resize([cfg.resolution, cfg.resolution]),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                        ])

        fill_transform = A.Compose([
            # A.RandomCrop(width=256, height=256),
            A.Resize(height=cfg.resolution, width=cfg.resolution, always_apply=True),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.RandomRain(p=0.5),
            A.RandomFog(p=0.5),
            # transforms.ToTensor(),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        ds = ObservationsDataset(df, root=os.path.join(ds_root, cfg.dataset.images),
                                label_col=cfg.dataset.label_col, filename_col=cfg.dataset.filename_col,
                                window_size=2, resolution=cfg.resolution,
                                transform=transform, fill_transform=None)
        dst, dsv = ds.split(train_perc=cfg.dataset.train_perc)
        return ds, dst, dsv
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
@hydra.main(config_path="config", config_name="supr_hresnet50.yaml")
# @hydra.main(config_path="config", config_name="supr_hresnet101.yaml")
# @hydra.main(config_path="config", config_name="supr_vitae.yaml")
# @hydra.main(config_path="config", config_name="supr_hefficientnet_b4.yaml")
# @hydra.main(config_path="config", config_name="supr_hcct_14_7x2_224.yaml")
# @hydra.main(config_path="config", config_name="supr_hdensenet.yaml")
# @hydra.main(config_path="config", config_name="supr_hefficientnet_b4.yaml")
# @hydra.main(config_path="config", config_name="supr_obs_hefficientnet_b4.yaml")
# @hydra.main(config_path="config", config_name="supr_obs_hresnet50.yaml")
def execute_training(cfg: DictConfig):
    exp_name = get_exp_name(cfg)
    original_path = CODE_ROOT # hydra.utils.get_original_cwd()

    real_batch_size = cfg.batch_size
    accumulation_steps = 32  # desired_batch_size // real_batch_size
    ds, dst, dsv = get_dataset(original_path=original_path, cfg=cfg)
    loader = torch.utils.data.DataLoader(ds, batch_size=real_batch_size, shuffle=True)


    model = get_model(cfg=cfg, original_path=original_path, num_classes=ds.class_size)
    # engine = get_engine(cfg=cfg, ds=ds, loader=loader, loader_val=None, model=model, class_dim=ds.class_size, epochs=cfg.epochs)

    from torch_lr_finder import LRFinder

        # Beware of the `batch_size` used by `DataLoader`
    # loader = torch.utils.data.DataLoader(dst, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
    #                                      num_workers=cfg.num_workers, persistent_workers=True, pin_memory=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=cfg.lr,
                                   weight_decay=cfg.weight_decay, eps=1e-4)

    # (Optional) With this setting, `amp.scale_loss()` will be adopted automatically.
    from apex import amp

    model, optimizer = amp.initialize(model.to("cuda"), optimizer, opt_level='O1')

    lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
    lr_finder.range_test(loader, end_lr=10, num_iter=100, step_mode="exp", accumulation_steps=accumulation_steps)
    lr_finder.plot()
    import matplotlib.pyplot as plt
    plt.savefig(CODE_ROOT + 'nnloss.png')
    lr_finder.reset()

if __name__ == "__main__":
    execute_training()