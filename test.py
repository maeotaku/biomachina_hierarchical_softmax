import os
import pickle
import warnings

import hydra
import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, StochasticWeightAveraging
from torch.utils.data import DataLoader
from torchvision import transforms

from our_datasets import PlatCLEFSimCLR, PlantCLEF2022Supr
from engines import SimCLREngine, SuprEngine
from models.factory import create_model

from summary import *

warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)



def get_engine(cfg, loader, loader_val, model, class_dim):
    if cfg.engine.name == "simclr":
        return SimCLREngine(model=model, loader=loader, loader_val=loader_val, cfg=cfg)
    else:
        return SuprEngine(model=model, loader=loader, loader_val=loader_val, cfg=cfg, class_dim=class_dim)


def get_model(cfg, original_path, num_classes):
    model_cfg = OmegaConf.to_container(cfg.model)
    model_cfg["num_classes"] = num_classes
    pretrained = cfg.pretrained_model
    checkpoint_dir = get_full_path(original_path, cfg.model_checkpoints)
    model_check_point = os.path.join(checkpoint_dir, cfg.pretrained_model_point)
    return create_model(model_name=cfg.model.name, pretrained=pretrained, checkpoint_path=model_check_point,
                        **model_cfg)


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
        return ds


# @hydra.main(config_path="config", config_name="simclr_vit.yaml")
#@hydra.main(config_path="config", config_name="supr_hresnet50.yaml")
# @hydra.main(config_path="config", config_name="supr_hresnet101.yaml")
# @hydra.main(config_path="config", config_name="supr_vitae.yaml")
# @hydra.main(config_path="config", config_name="supr_hefficientnet_b4.yaml")
@hydra.main(config_path="config", config_name="supr_hefficientnet_b0.yaml")
# @hydra.main(config_path="config", config_name="supr_hcct_14_7x2_224.yaml")
# @hydra.main(config_path="config", config_name="supr_hdensenet.yaml")
def run(cfg: DictConfig):
    exp_name = get_exp_name(cfg)
    original_path = hydra.utils.get_original_cwd()
    ds = get_dataset(original_path=original_path, cfg=cfg)
    loader = torch.utils.data.DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
                                         num_workers=cfg.num_workers, persistent_workers=True, pin_memory=True)
    import os
    import numpy as np
    from tqdm import tqdm
    from PIL import Image

    cont = 0

    import threading, time, random
    from queue import Queue

    jobs = Queue()

    def compress(open_path, basepath, filename):
        foo = Image.open(os.path.join(open_path, filename))
        foo = foo.resize((224, 224), Image.ANTIALIAS)
        foo.save(os.path.join(basepath, filename), quality=95, torch_optimizer=True)

    basepath = f"d:/compressed"
    open_path = f"C:/Users\maeot/Documents\code/biomachina/trusted/images"
    for x, y, filename, classid in tqdm(loader):
        for i in range(x.shape[0]):
            classpath = os.path.join(basepath, str(classid[i].item()))
            if not os.path.exists(classpath):
                os.mkdir(classpath)
            # torch.save( x[i], os.path.join(basepath, f"{filename[i]}.pth"))
            # np.savez_compressed(, x[i].numpy())
            compress(open_path, basepath, filename[i])

        cont+=1




if __name__ == "__main__":
    run()
