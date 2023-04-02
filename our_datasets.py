import pickle
import random

import pandas as pd
from torchvision import datasets, transforms
from tqdm import tqdm

tqdm.pandas()
import io
import os
import sys
from collections import Counter
from os.path import exists

import numpy as np
import torch
import torchvision
from lxml import etree
from PIL import Image, ImageFile
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

def is_valid(filename):
    img = Image.open(filename)
    try:
        img.verify()
        return True
    except Exception:
        return False

def has_bytes(filename):
    return os.stat(filename).st_size > 0

def is_image_file(filename):
    res = has_file_allowed_extension(filename, IMG_EXTENSIONS) and os.path.isfile(filename) and has_bytes(filename) # and is_valid(filename)
    return res

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def has_file_allowed_extension(filename, extensions):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)
    
def is_image_file_from_df(root, filename):
    return is_image_file(os.path.join(root, filename))

def check_file_existence(df : pd.DataFrame, root : str, label_col : str, filename_col : str):
    df["file_exists"] = df.progress_apply(lambda x : is_image_file_from_df(root, x[filename_col]), axis = 1)
    df = df[df["file_exists"]==True]
    return df

class LifeCLEFDataset(Dataset):
    """Defines an abstract class that knows how to consume plantclef 2002 dataset."""

    def split(self, train_perc : float = 0.8):
        train_indices, val_indices = train_test_split(list(range(len(self.targets))), test_size=1.0-train_perc,
                                                      stratify=self.targets)
        train_dataset = torch.utils.data.Subset(self, train_indices)
        val_dataset = torch.utils.data.Subset(self, val_indices)
        return train_dataset, val_dataset
    
    def generate_classes_dict(self, df, label_col):
        """Generates a list and inverted list of classes and their indexes"""

        all_classes = set(pd.DataFrame(df[label_col].value_counts().keys()).iloc[:, 0])
        for v, k in enumerate(all_classes):
            self.class_dict[k] = v
            self.inv_class_dict[v] = k
        
    
    def target_transform(self, class_text):
        """Transforms text into a class id"""
        return self.class_dict[class_text]
    
    def clean_dataset(self, df, root, label_col, filename_col):
        df = check_file_existence(df, root, label_col, filename_col)
        return df
    
    def make_samples(self, df, label_col, filename_col):
        """Converts a 2 column dataframe into a list of tuples with the samples"""
        df['count'] = df.groupby([label_col])[label_col].transform('count')
        # Drop indexes for count value == 1
        df = df.drop(df[df['count'] == 1].index)
        return list(df[[filename_col, label_col]].to_records(index=False)), df

    def __init__(self, df, root : str, label_col : str, filename_col : str, loader=default_loader, transform=None, class_dict=None, inv_class_dict=None):
        self.root = root
        self.label_col = label_col

        print("Cleaning dataset...")
        df = self.clean_dataset(df, root, label_col, filename_col)
        print("Making samples....")
        self.samples, self.df = self.make_samples(df, label_col, filename_col)
        if len(self) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(IMG_EXTENSIONS)))
        self.targets = [ sample[1] for sample in self.samples ]
        self.inv_class_dict = inv_class_dict # if inv_class_dict is not None else {}
        self.class_dict = class_dict #if class_dict is not None else  {}
        
        self.loader = loader
        self.transform_x = transform

    def __len__(self):
        return len(self.samples)
    
    @property
    def class_size(self):
        return len(self.class_dict.keys())

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform_x.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        fmt_str += f'Label: {self.label_col} - {self.class_size}'
        fmt_str += f'Size: {self.df.shape}'
        # tmp = '    Target Transforms (if any): '
        # fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        filename, target = self.samples[index]
        sample = self.loader(os.path.join(self.root, filename))
        if self.transform_x is not None:
            sample = self.transform_x(sample)
        target_t = self.target_transform(target)

        sample = np.array(sample)

        return sample, target_t

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

class FungiDataModule(LightningDataModule):

    FILENAME_TRAIN = "fungi_ds_train_cache.pickle"
    FILENAME_VAL = "fungi_ds_val_cache.pickle"

    def __init__(self, original_path, cfg):
        super().__init__()

        columns = ["class_id", "image_path"]
        self.ds_root = get_full_path(original_path, cfg.dataset.path)
        self.df = pd.read_csv(os.path.join(self.ds_root, cfg.dataset.csv), sep=",", usecols=columns)
        self.df["class_id"] = self.df["class_id"].apply(lambda x : x + 1)
        self.ds_root_val = get_full_path(original_path, cfg.dataset.path_val)
        self.df_val = pd.read_csv(os.path.join(self.ds_root_val, cfg.dataset.csv_val), sep=",", usecols=columns)
        self.df_val["class_id"] = self.df_val["class_id"].apply(lambda x : x + 1)

        self.all_classes = set(pd.DataFrame(self.df["class_id"].value_counts().keys()).iloc[:, 0]).union(set(pd.DataFrame(self.df_val["class_id"].value_counts().keys()).iloc[:, 0]))

        self.class_dict = {}
        self.inv_class_dict = {}
        for v, k in enumerate(self.all_classes):
            self.class_dict[k] = v
            self.inv_class_dict[v] = k

        

        self.transform_train = transforms.Compose([ #transforms.AutoAugment(AutoAugmentPolicy.IMAGENET),
                                        transforms.Resize([cfg.resolution, cfg.resolution]),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                        ])
        # self.transform_val = transforms.Compose([
        #                                 # transforms.AutoAugment(AutoAugmentPolicy.IMAGENET),
        #                                 transforms.Resize([cfg.resolution_test, cfg.resolution_test]),
        #                                 transforms.ToTensor(),
        #                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #                                 ])

        self.batch_size = cfg.batch_size
        self.num_workers = cfg.num_workers
        self.cfg = cfg
        self.original_path = original_path

    @property
    def class_size(self):
        return len(self.class_dict)
       
    def setup(self, stage=None):

        path = os.path.join(self.original_path, self.FILENAME_VAL)
        if exists(path):
            self.val_dataset = pickle.load(open(path, "rb"))
            print("Dataset cache loaded.")
        else:
            self.val_dataset = LifeCLEFDataset(self.df_val, root=os.path.join(self.ds_root_val, self.cfg.dataset.images_val),
                                   label_col=self.cfg.dataset.label_col, filename_col=self.cfg.dataset.filename_col,
                                   transform=self.transform_train, class_dict=self.class_dict, inv_class_dict=self.inv_class_dict)
            pickle.dump(self.val_dataset, open(path, "wb"))

        path = os.path.join(self.original_path, self.FILENAME_TRAIN)
        if exists(path):
            self.train_dataset = pickle.load(open(path, "rb"))
            print("Dataset cache loaded.")
        else:
            self.train_dataset = LifeCLEFDataset(self.df, root=os.path.join(self.ds_root, self.cfg.dataset.images),
                                    label_col=self.cfg.dataset.label_col, filename_col=self.cfg.dataset.filename_col,
                                    transform=self.transform_train, class_dict=self.class_dict, inv_class_dict=self.inv_class_dict)
            pickle.dump(self.train_dataset, open(path, "wb"))

        print(self.train_dataset)
        print(self.val_dataset)

        train_split1, train_split2 = self.train_dataset.split(train_perc = self.cfg.dataset.train_perc)
        val_split1, val_split2 = self.val_dataset.split(train_perc = self.cfg.dataset.train_perc)
        self.train_dataset = torch.utils.data.ConcatDataset([train_split1, val_split1])
        self.val_dataset = torch.utils.data.ConcatDataset([train_split2, val_split2])




    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
        )


class PlantDataModule(LightningDataModule):

    FILENAME_TRAIN = "plant_ds_train_cache.pickle"

    def __init__(self, original_path, cfg):
        super().__init__()

        columns = ["classid", "image_path"]
        self.ds_root = get_full_path(original_path, cfg.dataset.path)
        self.df = pd.read_csv(os.path.join(self.ds_root, cfg.dataset.csv), sep=";", usecols=columns)

        self.all_classes = set(
                                pd.DataFrame(self.df["classid"].value_counts().keys()).iloc[:, 0]
                            )

        self.class_dict = {}
        self.inv_class_dict = {}
        for v, k in enumerate(self.all_classes):
            self.class_dict[k] = v
            self.inv_class_dict[v] = k

        self.batch_size = cfg.batch_size
        self.num_workers = cfg.num_workers
        self.cfg = cfg
        self.original_path = original_path

    @property
    def class_size(self):
        return len(self.class_dict)
       
    def setup(self, stage=None):
        path = os.path.join(self.original_path, self.FILENAME_TRAIN)
        if exists(path):
            self.dataset = pickle.load(open(path, "rb"))
            print("Dataset cache loaded.")
        else:
            self.dataset = LifeCLEFDataset(self.df, root=os.path.join(self.ds_root, self.cfg.dataset.images),
                                    label_col=self.cfg.dataset.label_col, filename_col=self.cfg.dataset.filename_col,
                                    transform=self.transform_train, class_dict=self.class_dict, inv_class_dict=self.inv_class_dict)
            pickle.dump(self.train_dataset, open(path, "wb"))

        self.train_dataset, self.val_dataset = self.dataset.split(train_perc = 0.8)

    def train_dataloader(self):

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
        )


class PlantAndFungiDataModule(LightningDataModule):

    FILENAME_PLANT_TRAIN = "plant_ds_train_cache.pickle"
    FILENAME_FUNGI_TRAIN = "fungi_ds_train_cache.pickle"
    FILENAME_FUNGI_VAL = "fungi_ds_val_cache.pickle"

    def __init__(self, original_path, cfg):
        super().__init__()

        fungi_columns = ["class_id", "image_path"]
        self.fungi_ds_root = get_full_path(original_path, cfg.dataset.fungi_path)
        self.fungi_df = pd.read_csv(os.path.join(self.fungi_ds_root, cfg.dataset.fungi_csv), sep=",", usecols=fungi_columns)
        self.fungi_df["class_id"] = self.fungi_df["class_id"].apply(lambda x : f"F_{x}")

        self.fungi_ds_root_val = get_full_path(original_path, cfg.dataset.fungi_path_val)
        self.fungi_df_val = pd.read_csv(os.path.join(self.fungi_ds_root_val, cfg.dataset.fungi_csv_val), sep=",", usecols=fungi_columns)
        self.fungi_df_val["class_id"] = self.fungi_df_val["class_id"].apply(lambda x : f"F_{x}")

        plant_columns = ["classid", "image_path"]
        self.plant_ds_root = get_full_path(original_path, cfg.dataset.plant_path)
        self.plant_df = pd.read_csv(os.path.join(self.plant_ds_root, cfg.dataset.plant_csv), sep=";", usecols=plant_columns)
        self.plant_df["classid"] = self.plant_df["classid"].apply(lambda x : f"P_{x}")

        self.all_classes = set(
                                pd.DataFrame(self.fungi_df["class_id"].value_counts().keys()).iloc[:, 0]
                            ).union(
                                set(pd.DataFrame(self.fungi_df_val["class_id"].value_counts().keys()).iloc[:, 0])
                            ).union(
                                set(pd.DataFrame(self.plant_df["classid"].value_counts().keys()).iloc[:, 0])
                            )

        self.class_dict = {}
        self.inv_class_dict = {}
        for v, k in enumerate(self.all_classes):
            self.class_dict[k] = v
            self.inv_class_dict[v] = k

        #assign indexes of each type of data to calculate metrics
        self.fungi_indexes = []
        self.plant_indexes = []
        for k, v in self.class_dict.items():
            if "P" in k:
                self.plant_indexes.append(v)
            else:
                self.fungi_indexes.append(v)
        self.fungi_indexes = torch.LongTensor(self.fungi_indexes)
        self.plant_indexes = torch.LongTensor(self.plant_indexes)
        print(self.fungi_indexes.shape)
        print(self.plant_indexes.shape)

        self.transform_train = transforms.Compose([ #transforms.AutoAugment(AutoAugmentPolicy.IMAGENET),
                                        transforms.Resize([cfg.resolution, cfg.resolution]),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                        ])
        # self.transform_val = transforms.Compose([
        #                                 transforms.AutoAugment(AutoAugmentPolicy.IMAGENET),
        #                                 transforms.Resize([cfg.resolution_test, cfg.resolution_test]),
        #                                 transforms.ToTensor(),
        #                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #                                 ])

        self.batch_size = cfg.batch_size
        self.num_workers = cfg.num_workers
        self.cfg = cfg
        self.original_path = original_path

    @property
    def class_size(self):
        return len(self.class_dict)

    def setup(self, stage=None):
        #FUNGI
        path = os.path.join(self.original_path, self.FILENAME_FUNGI_TRAIN)
        if exists(path):
            self.fungi_train_dataset = pickle.load(open(path, "rb"))
            print("Dataset cache loaded.")
        else:
            self.fungi_train_dataset = LifeCLEFDataset(self.fungi_df, root=os.path.join(self.fungi_ds_root, self.cfg.dataset.fungi_images),
                                    label_col=self.cfg.dataset.fungi_label_col, filename_col=self.cfg.dataset.fungi_filename_col,
                                    transform=self.transform_train, class_dict=self.class_dict, inv_class_dict=self.inv_class_dict)
            pickle.dump(self.fungi_train_dataset, open(path, "wb"))


        path = os.path.join(self.original_path, self.FILENAME_FUNGI_VAL)
        if exists(path):
            self.fungi_val_dataset = pickle.load(open(path, "rb"))
            print("Dataset cache loaded.")
        else:
            self.fungi_val_dataset = LifeCLEFDataset(self.fungi_df_val, root=os.path.join(self.fungi_ds_root_val, self.cfg.dataset.fungi_images_val),
                                   label_col=self.cfg.dataset.fungi_label_col, filename_col=self.cfg.dataset.fungi_filename_col,
                                   transform=self.transform_train, class_dict=self.class_dict, inv_class_dict=self.inv_class_dict)
            pickle.dump(self.fungi_val_dataset, open(path, "wb"))

        #PLANT
        path = os.path.join(self.original_path, self.FILENAME_PLANT_TRAIN)
        if exists(path):
            self.plant_train_dataset = pickle.load(open(path, "rb"))
            print("Dataset cache loaded.")
        else:
            self.plant_train_dataset = LifeCLEFDataset(self.plant_df, root=os.path.join(self.plant_ds_root, self.cfg.dataset.plant_images),
                                    label_col=self.cfg.dataset.plant_label_col, filename_col=self.cfg.dataset.plant_filename_col,
                                    transform=self.transform_train, class_dict=self.class_dict, inv_class_dict=self.inv_class_dict)
            pickle.dump(self.plant_train_dataset, open(path, "wb"))

        print(f"Total classes: {self.class_size}")

        fungi_train_split1, fungi_train_split2 = self.fungi_train_dataset.split(train_perc = self.cfg.dataset.train_perc)
        fungi_val_split1, fungi_val_split2 = self.fungi_val_dataset.split(train_perc = self.cfg.dataset.train_perc)
        plant_train_split1, plant_train_split2 = self.plant_train_dataset.split(train_perc = self.cfg.dataset.train_perc)
        self.train_dataset = torch.utils.data.ConcatDataset([fungi_train_split1, fungi_val_split1, plant_train_split1])
        self.val_dataset = torch.utils.data.ConcatDataset([fungi_train_split2, fungi_val_split2, plant_train_split2])

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
        )

# def simclr_transform(size, s=1):
#     """Return a set of data augmentation transformations as described in the SimCLR paper."""
#     color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
#     data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
#                                           transforms.RandomHorizontalFlip(),
#                                           transforms.RandomApply([color_jitter], p=0.8),
#                                           transforms.RandomGrayscale(p=0.2),
#                                           GaussianBlur(kernel_size=int(0.1 * size)),
#                                           transforms.ToTensor()])
#     return data_transforms

# class ContrastiveLearningViewGenerator(object):
#     """Take two random crops of one image as the query and key."""

#     def __init__(self, base_transform, n_views=2):
#         self.base_transform = base_transform
#         self.n_views = n_views

#     def __call__(self, x):
#         return [self.base_transform(x) for i in range(self.n_views)]


   
# class PlatCLEFSimCLR(LifeCLEFDataset):
#     """Data augmented dataset using plantclef2022 for self supervision"""
    
#     def __init__(self, df, root : str, label_col : str, filename_col : str, loader=default_loader, size=224, n_views=2):
#         transform = ContrastiveLearningViewGenerator(simclr_transform(size), n_views)
#         super().__init__(df=df, root=root, label_col=label_col, filename_col=filename_col, loader=loader, transform=transform)
    
#     def __getitem__(self, index):
#         """
#         Args:
#             index (int): Index
#         Returns:
#             tuple: (sample, target) where target is class_index of the target class.
#         """
#         filename, target = self.samples[index]
#         # samples are a list of n_views=2, each with a tensor of size (B, C, H, W)
#         sample = self.loader(os.path.join(self.root, filename))
#         if self.transform_x is not None:
#             sample = self.transform_x(sample)
#         target_t = self.target_transform(target)

#         target_t = torch.tensor(target_t)
#         return sample, target_t
    

# class ObservationsDataset(torch.utils.data.Dataset):
#     def __init__(self, df, root: str, label_col: str = "classid", filename_col: str = "image_path",
#                  obs_col: str = "gbif_occurrence_id", window_size=4, channels=3, resolution=224, loader=default_loader,
#                  transform=None, fill_transform=None):

#         self.transform = transform
#         self.fill_transform = fill_transform
#         self.window_size = window_size
#         self.channels = channels
#         self.resolution = resolution
#         self.label_col = label_col
#         self.obs_col = obs_col
#         self.filename_col = filename_col
#         self.root = root
#         self.loader = loader

#         df["Index"] = df.index
#         df[obs_col].fillna(df.Index, inplace=True)

#         all_classes = list(df[self.label_col].unique())
#         self.class_dict = {k: v for v, k in enumerate(all_classes)}
#         self.inv_class_dict = {v: k for v, k in enumerate(all_classes)}

#         self.obs = df.groupby(obs_col)
#         self.restart_windows()

#     def split(self, train_perc: float = 0.8, keep_split : bool = True):
#         if not keep_split or self.train_indices is None: 
#             self.train_indices, self.val_indices = train_test_split(list(range(len(self.targets))), test_size=1.0 - train_perc)
#                                                       # stratify=self.targets)
#         train_dataset = torch.utils.data.Subset(self, self.train_indices)
#         val_dataset = torch.utils.data.Subset(self, self.val_indices)
#         return train_dataset, val_dataset

#     def restart_windows(self):
#         self.observations = []
#         self.targets = []
#         self.obs.progress_apply(lambda ob: self.ob(ob))

#     def ob(self, images):
#         classid = images.iloc[0][self.label_col]
#         image_paths = images[self.filename_col].tolist()
#         random.shuffle(image_paths)
#         windows = [image_paths[x:x + self.window_size] for x in range(0, len(image_paths), self.window_size)]
#         self.observations += windows
#         self.targets += [self.class_dict[classid]] * len(windows)

#     def __len__(self):
#         return len(self.observations)

#     def target_transform(self, class_text):
#         """Transforms text into a class id"""
#         return self.class_dict[class_text]

#     def __getitem__(self, index):
#         ob = self.observations[index]
#         window_count = len(ob)
#         window = torch.zeros(self.window_size, self.channels, self.resolution, self.resolution)
#         for i in range(self.window_size):
#             if i < window_count:
#                 full_path = os.path.join(self.root, ob[i])
#                 image = self.loader(full_path)
#                 if self.transform is not None:
#                     image = self.transform(image)
#                 window[i] = image
#             else:
#                 if self.fill_transform is not None:
#                     full_path = os.path.join(self.root, ob[random.randint(0, window_count - 1)])
#                     image = self.loader(full_path)
#                     image = np.array(image)
#                     image = self.fill_transform(image=image)['image']
#                     image = torch.tensor(image.transpose(2, 0, 1))
#                     window[i] = image
#             # else:
#             #     window.append(torch.zeros(3, 224, 224))

#         # window = torch.cat(window, dim=0)
#         target = self.targets[index]
#         return window, target

#     @property
#     def class_size(self):
#         return len(self.class_dict.keys())

    

def get_dataset(original_path, cfg):
    if cfg.dataset.name == "fungi":
        return FungiDataModule(original_path, cfg=cfg)
    elif cfg.dataset.name == "fungi_and_plant":
        return PlantAndFungiDataModule(original_path, cfg=cfg)
    else: #defaul plants
        return PlantDataModule(original_path, cfg=cfg)