from torchvision.transforms import transforms
from gaussian_blur import GaussianBlur
from torchvision import transforms, datasets
import sys
import io
import os
from lxml import etree

import torch
from torch.utils.data.dataset import Dataset
import torchvision
import numpy as np
from PIL import Image

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

def is_image_file(filename):
    return has_file_allowed_extension(filename, IMG_EXTENSIONS) and os.path.isfile(filename)

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

def check_file_existence(df, root : str, label_col : str, filename_col : str):
        df["file_exists"] = df.apply(lambda x : is_image_file_from_df(root, x[filename_col]), axis = 1)
        df = df[df["file_exists"]==True]
        return df

class PlantCLEF2022_Dataset(Dataset):
    """Defines an abstract class that knows how to consume plantclef 2002 dataset."""
    
    def generate_classes_dict(self, df, label_col):
        """Generates a list and inverted list of classes and their indexes"""
        all_classes = list(df[label_col].unique())
        self.class_dict = {k: v for v, k in enumerate(all_classes)}
        self.inv_class_dict = { v: k for v, k in enumerate(all_classes)}
        
    
    def target_transform(self, class_text):
        """Transforms text into a class id"""
        return self.class_dict[class_text]
    
    def clean_dataset(self, df, root, label_col, filename_col):
        return check_file_existence(df, root, label_col, filename_col)
    
    def make_samples(self, df, label_col, filename_col):
        """Converts a 2 column dataframe into a list of tuples with the samples"""
        return list(df[[filename_col, label_col]].to_records(index=False))

    def __init__(self, df, root : str, label_col : str, filename_col : str, loader=default_loader, transform=None):
        self.root = root
        df = self.clean_dataset(df, root, label_col, filename_col)
        self.generate_classes_dict(df, label_col)
        self.samples = self.make_samples(df, label_col, filename_col)
        if len(self) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(IMG_EXTENSIONS)))
        self.loader = loader

        # self.targets = [s[1] for s in self.samples]

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
        # tmp = '    Target Transforms (if any): '
        # fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
    

def simclr_transform(size, s=1):
    """Return a set of data augmentation transformations as described in the SimCLR paper."""
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomApply([color_jitter], p=0.8),
                                          transforms.RandomGrayscale(p=0.2),
                                          GaussianBlur(kernel_size=int(0.1 * size)),
                                          transforms.ToTensor()])
    return data_transforms

class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]


   
class PlatCLEFSimCLR(PlantCLEF2022_Dataset):
    """Data augmented dataset using plantclef2022 for self supervision"""
    
    def __init__(self, df, root : str, label_col : str, filename_col : str, loader=default_loader, size=224, n_views=2):
        transform = ContrastiveLearningViewGenerator(simclr_transform(size), n_views)
        super().__init__(df=df, root=root, label_col=label_col, filename_col=filename_col, loader=loader, transform=transform)
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        filename, target = self.samples[index]
        # samples are a list of n_views=2, each with a tensor of size (B, C, H, W)
        sample = self.loader(os.path.join(self.root, filename))
        if self.transform_x is not None:
            sample = self.transform_x(sample)
        target = self.target_transform(target)

        target = torch.tensor(target)
        return sample, target
    
class PlantCLEF2022Supr(PlantCLEF2022_Dataset):
    """Dataset used to train a classifier with plantclef2022"""
    
    def __init__(self, df, root : str, label_col : str, filename_col : str, loader=default_loader, transform=None):
        super().__init__(df=df, root=root, label_col=label_col, filename_col=filename_col, loader=loader, transform=transform)
    
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
        target = self.target_transform(target)

        sample = np.array(sample)
        return sample, target