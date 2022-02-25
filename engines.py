import logging
import os
import sys

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from metrics import accuracy

import pytorch_lightning as pl


class SimCLREngine(pl.LightningModule):
    
    def __init__(self, model, loader, args):
        super().__init__()
        self.model = model
        self.args = args
        self.loader = loader
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)
        
    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), self.args['lr'], weight_decay=self.args['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(self.loader), eta_min=0,
                                                               last_epoch=-1)
        return [optimizer], [scheduler]

    
    def info_nce_loss(self, features):

        labels = torch.cat([torch.arange(self.args['batch_size']) for i in range(self.args['n_views'])], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool) #.to(device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        logits = logits / self.args['temperature']
        return logits, labels

    def training_step(self, train_batch, batch_idx):
        x, _ = train_batch
        x = torch.cat(x, dim=0)
        features = self.model(x)
        logits, labels = self.info_nce_loss(features)
        loss = self.criterion(logits, labels)
        top1, top5 = accuracy(logits, labels, topk=(1, 5))
        self.log('selfsupr_train_acc_top1', top1[0])
        self.log('selfsupr_train_acc_top5', top5[0])
        self.log("selfsupr_train_loss", loss)
        return loss



class SuprEngine(pl.LightningModule):
    
    def __init__(self, model, loader, args):
        super().__init__()
        self.model = model
        self.args = args
        self.loader = loader
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)
        
    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), self.args['lr'], weight_decay=self.args['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(self.loader), eta_min=0,
                                                               last_epoch=-1)
        return [optimizer], [scheduler]


    def training_step(self, train_batch, batch_idx):
        x, labels = train_batch
        logits = self.model(x)
        loss = self.criterion(logits, labels)
        top1, top5 = accuracy(logits, labels, topk=(1, 5))
        self.log('supr_train_acc_top1', top1[0])
        self.log('supr_train_acc_top5', top5[0])
        self.log("supr_train_loss", loss)
        return loss

#     def validation_step(self, val_batch, batch_idx):
#         pass
# # 		x, y = val_batch
# # 		x = x.view(x.size(0), -1)
# # 		z = self.encoder(x)
# # 		x_hat = self.decoder(z)
# # 		loss = F.mse_loss(x_hat, x)
# # 		self.log('val_loss', loss)



