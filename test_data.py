# Others
import os
import random

# Pandas
import pandas as pd
# Pytorch
import torch
# Torchvision for CV
import torchvision
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
tqdm.pandas()

# Pickle
# Scipy
# Sklearn
# warnings.filterwarnings('ignore')

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler
import numpy as np

import itertools

#
# class ObservationsDataset(torch.utils.data.Dataset):
#     def __init__(self, transform, window_size=4):
#         self.basePath = "trusted/images/"
#         self.fileName = "trusted/PlantCLEF2022_trusted_training_metadata.csv"
#         self.transform = transform
#         self.window_size = window_size
#
#         df = pd.read_csv(self.fileName, sep=';', usecols=["classid", "image_path", "gbif_occurrence_id"])[:1000]
#         df["Index"] = df.index
#         df.gbif_occurrence_id.fillna(df.Index, inplace=True)
#
#         all_classes = list(df.classid.unique())
#         self.class_dict = {k: v for v, k in enumerate(all_classes)}
#         self.inv_class_dict = {v: k for v, k in enumerate(all_classes)}
#
#         self.obs = df.groupby('gbif_occurrence_id')
#         self.restart_windows()
#
#     def restart_windows(self):
#         self.observations = []
#         self.targets = []
#         self.obs.progress_apply(lambda ob: self.ob(ob))
#
#     def ob(self, images):
#         classid = images.iloc[0]["classid"]
#         image_paths = images["image_path"].tolist()
#         random.shuffle(image_paths)
#         windows = [image_paths[x:x + self.window_size] for x in range(0, len(image_paths), self.window_size)]
#         self.observations += windows
#         self.targets += [self.class_dict[classid]] * len(windows)
#
#     def __len__(self):
#         return len(self.observations)
#
#     def __getitem__(self, index):
#         ob = self.observations[index]
#         window_count = len(ob)
#         window = torch.zeros(self.window_size, 3, 224, 224)
#         for i in range(self.window_size):
#             if i < window_count:
#                 full_path = os.path.join(self.basePath, ob[i])
#                 image = Image.open(full_path).convert('RGB')
#                 image = self.transform(image)
#                 window[i] = image
#             # else:
#             #     window.append(torch.zeros(3, 224, 224))
#
#         # window = torch.cat(window, dim=0)
#         return window, self.targets[index]
#
# obs = {
#     1 : ["1.jpg"],
#     2 : ["2.jpg"],
#     3 : ["3.jpg"],
#     4 : ["4.jpg"],
#     5 : ["5.jpg"],
#     11 : ["1.1.jpg", "1.2.jpg"],
#     22 : ["2.1.jpg", "2.2.jpg"],
#     33 : ["3.1.jpg", "3.2.jpg"],
#     44 : ["4.1.jpg", "4.2.jpg"],
#     55 : ["5.1.jpg", "5.2.jpg"],
#     111 : ["11.1.jpg", "11.2.jpg", "11.3.jpg"],
#     222 : ["221.jpg", "22.2.jpg", "22.3.jpg"],
#     1111 : ["111.1.jpg", "111.2.jpg", "111.3.jpg", "111.4.jpg"],
#     2222 : ["222.1.jpg", "222.2.jpg", "222.3.jpg", "222.4.jpg"],
#     11111 : ["1111.1.jpg", "2222.1.jpg", "2222.2.jpg", "3333.3.jpg", "4444.1.jpg"],
# }
#
# obs_windows = {
#     1 : [ 1, 2, 3, 4, 5],
#     2 : [ 11, 22, 33, 44, 55],
#     3 : [ 111, 222],
#     4:  [ 1111, 2222],
#     5:  [ 11111]
# }
#
# obs_tgts = {
#     1 : [ 1, 1, 3, 1, 5],
#     2 : [ 2, 1, 2, 1, 5],
#     3 : [ 1, 3],
#     4:  [ 3, 4],
#     5:  [ 5 ]
# }
#
# def get_len(obs_windows):
#     cont = 0
#     for k, v in obs_windows.items():
#         cont+= (k * len(v))
#     return cont
#
# class BalancedBatchSampler(BatchSampler):
#     def __init__(self, max_batch_size, obs_windows, obs_tgts, obs):
#         # super(BalancedBatchSampler, self).__init__()
#         self.max_batch_size = max_batch_size
#         self.obs = obs
#         self.obs_windows = obs_windows
#         self.obs_tgts = obs_tgts
#         self.n_dataset = get_len(obs)
#
#     def _delete_from_obs(self):
#         pass
#
#     def _calc_batched_seq_size(self, curr_seq):
#         return self.max_batch_size % curr_seq
#
#     def _flatten(self, window):
#         return list(itertools.chain(*window))
#
#     def __iter__(self):
#         self.count = 0
#         while self.count < self.n_dataset:
#             indices = []
#             for k, windows in self.obs_windows.items():
#                 batched_seq_size = self._calc_batched_seq_size(k)
#                 if batched_seq_size <= len(windows):
#                     indices += self._flatten(windows[:, batched_seq_size])
#                     # del self.obs_windows[k][:, batched_seq_size] #delete used images
#                 else: #if there are not enough windows, use all of them and delete the windows key
#                     indices += self._flatten(windows)
#                     # del self.obs_windows[k]  # delete the whole key
#             yield indices
#             self.count += len(indices)
#
#     def _calc_total_batches(self):
#         total = 0
#         for k, windows in self.obs_windows.items():
#             batched_seq_size = self._calc_batched_seq_size(k)
#             total += batched_seq_size
#             total += (len(windows) - batched_seq_size)
#         return total
#
#     def __len__(self):
#         return self._calc_total_batches()




transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST(root='./dataset',
                               train=True, download=True,
                               transform=transform)
print(train_dataset.train_data[0].shape)
print(train_dataset.train_labels.shape)

train_batch_sampler = BalancedBatchSampler(train_dataset.train_labels.numpy(), n_classes=10, n_samplers=10)
online_train_loader = DataLoader(train_dataset, batch_sampler=train_batch_sampler)

for idx, (x, label) in enumerate(online_train_loader):
    print(x.shape)
    print(label.shape)



root = "./"
df = pd.read_csv(os.path.join(root, "trusted/PlantCLEF2022_trusted_training_metadata.csv"), sep=';', usecols=[self.obs_col, self.label_col, self.filename_col])[:1000]


transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.3734], [0.2041])])
ds = ObservationsDataset(transform=transform)

loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=True, drop_last=True,
                                     num_workers=0)

import torch
import numpy as np
import torch
import torchvision.models as models
from torch import nn

class HierarchicalSoftmax(nn.Module):

    def __init__(self, ntokens, nhid, ntokens_per_class=None, **kwargs):
        super(HierarchicalSoftmax, self).__init__()

        # Parameters
        self.ntokens = ntokens
        self.nhid = nhid

        if ntokens_per_class is None:
            ntokens_per_class = int(np.ceil(np.sqrt(ntokens)))

        self.ntokens_per_class = ntokens_per_class

        self.nclasses = int(np.ceil(self.ntokens * 1. / self.ntokens_per_class))
        self.ntokens_actual = self.nclasses * self.ntokens_per_class

        self.layer_top_W = nn.Parameter(torch.FloatTensor(self.nhid, self.nclasses), requires_grad=True)
        # print(self.layer_top_W.shape)
        self.layer_top_b = nn.Parameter(torch.FloatTensor(self.nclasses), requires_grad=True)

        self.layer_bottom_W = nn.Parameter(torch.FloatTensor(self.nclasses, self.nhid, self.ntokens_per_class),
                                           requires_grad=True)
        # print(self.layer_bottom_W.shape)
        self.layer_bottom_b = nn.Parameter(torch.FloatTensor(self.nclasses, self.ntokens_per_class), requires_grad=True)

        self.softmax = nn.Softmax(dim=1)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.layer_top_W.data.uniform_(-initrange, initrange)
        self.layer_top_b.data.fill_(0)
        self.layer_bottom_W.data.uniform_(-initrange, initrange)
        self.layer_bottom_b.data.fill_(0)

    def _predict(self, inputs):
        batch_size, d = inputs.size()

        layer_top_logits = torch.matmul(inputs, self.layer_top_W) + self.layer_top_b
        layer_top_probs = self.softmax(layer_top_logits)

        label_position_top = torch.argmax(layer_top_probs, dim=1)

        layer_bottom_logits = torch.squeeze(
            torch.bmm(torch.unsqueeze(inputs, dim=1), self.layer_bottom_W[label_position_top]), dim=1) + \
                              self.layer_bottom_b[label_position_top]
        layer_bottom_probs = self.softmax(layer_bottom_logits)

        return torch.bmm(layer_top_probs.unsqueeze(2), layer_bottom_probs.unsqueeze(1)).flatten(start_dim=1)

    def forward(self, inputs, labels=None):
        if labels is None:
            return self._predict(inputs)
        batch_size, d = inputs.size()

        layer_top_logits = torch.matmul(inputs, self.layer_top_W) + self.layer_top_b
        layer_top_probs = self.softmax(layer_top_logits)

        label_position_top = (labels / self.ntokens_per_class).long()
        label_position_bottom = (labels % self.ntokens_per_class).long()

        # print(layer_top_probs.shape, label_position_top.shape)

        layer_bottom_logits = torch.squeeze(
            torch.bmm(torch.unsqueeze(inputs, dim=1), self.layer_bottom_W[label_position_top]), dim=1) + \
                              self.layer_bottom_b[label_position_top]
        layer_bottom_probs = self.softmax(layer_bottom_logits)

        target_probs = layer_top_probs[torch.arange(batch_size).long(), label_position_top] * layer_bottom_probs[
            torch.arange(batch_size).long(), label_position_bottom]

        # print(f"top {layer_top_probs.shape} {layer_top_probs}")
        # print(f"bottom {layer_bottom_probs.shape} {layer_bottom_probs}")
        top_indx = torch.argmax(layer_top_probs, dim=1)
        botton_indx = torch.argmax(layer_bottom_probs, dim=1)

        real_indx = (top_indx * self.ntokens_per_class) + botton_indx
        # print(top_indx, self.nclasses, botton_indx)
        # print(f"target {target_probs.shape} {target_probs}")

        # loss = -torch.mean(torch.log(target_probs.type(torch.float32) + 1e-3))
        loss = -torch.mean(torch.log(target_probs))
        with torch.no_grad():
            preds = torch.bmm(layer_top_probs.unsqueeze(2), layer_bottom_probs.unsqueeze(1)).flatten(start_dim=1)

        return loss, target_probs, layer_top_probs, layer_bottom_probs, top_indx, botton_indx, real_indx, preds


class ObservationTransformer(nn.Module):

    def __init__(self, encoder: nn.Module, nhead=10, feature_dim=1000, output_size=128):
        super().__init__()
        self.encoder = encoder
        self.head = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=nhead, batch_first=True)
        self.decoder = nn.Linear(feature_dim, output_size)
        self.feature_dim = feature_dim

    def forward(self, x):
        B, S, C, R, _ = x.shape
        x = x.reshape(-1, C, R, R)  # squeeze observations to (batch * sequence, channel, witdh, height)
        # size into a single dim for encoding as batch
        features = self.encoder(x)  # encode all images for B observations of S images each
        features = features.reshape(B, S, self.feature_dim)  # resize back to observation sizes (batch, sequence, )
        out = self.head(features)  # apply attention to teh features
        return self.decoder(out.mean(1))


class HObservationTransformer(nn.Module):

    def __init__(self, encoder: nn.Module, nhead=10, feature_dim=1000, classes=80000, ntokens_per_class=80):
        super().__init__()
        self.obs_transformer = ObservationTransformer(encoder=encoder, nhead=nhead, feature_dim=feature_dim,
                                                      output_size=128)
        self.hs = HierarchicalSoftmax(ntokens=classes, nhid=128, ntokens_per_class=ntokens_per_class)
        # print(self.hs.nclasses, self.hs.ntokens_per_class)

    def forward(self, x, y=None):
        assert len(x.shape) == 5, "Observation shape should be (batch, sequence, channels, witdh, height)"
        x = self.obs_transformer(x)
        return self.hs(x, y)


encoder = models.resnet50(pretrained=True)
# m = ObservationTransformer(encoder=encoder, nhead=10, feature_dim=1000, output_size=128)
hm = HObservationTransformer(encoder=encoder, nhead=10, feature_dim=1000, classes=80000, ntokens_per_class=80)









for x, y in loader:
    print(x.shape, y.shape)
    loss, target_probs, layer_top_probs, layer_bottom_probs, top_indx, botton_indx, real_indx, preds = hm(x, y)
    print(loss)