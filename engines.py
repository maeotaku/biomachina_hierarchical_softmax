import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch_optimizer as optim
import torchmetrics
import numpy as np

from metrics.mrr import MRR


class ExperimentEngine(pl.LightningModule):

    def __init__(self, model, cfg, epochs):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.epochs = epochs
        self.train_acc = torchmetrics.Accuracy()
        self.train_precision = torchmetrics.Precision()
        # self.batch_size = cfg.batch_size  # needed for automatic batch size calculation

    def forward(self, x):
        return self.model(x)

class SimCLREngine(ExperimentEngine):

    def __init__(self, model, loader, loader_val, cfg, epochs):
        super(SimCLREngine, self).__init__(model=model, loader=loader, loader_val=loader_val, cfg=cfg, epochs=epochs)
        self.criterion = torch.nn.CrossEntropyLoss()

    def configure_optimizers(self):
        optimizer = _get_optimizer(name=self.cfg.optimizer,
                                   params=filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.cfg.lr,
                                   weight_decay=self.cfg.weight_decay, momentum=self.cfg.momentum)
        scheduler = _get_scheduler(name=self.cfg.scheduler, optimizer=optimizer, epochs=self.epochs)

        return [optimizer], [scheduler]

    def info_nce_loss(self, features):
        labels = torch.cat([torch.arange(self.cfg.batch_size) for i in range(self.cfg.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        logits = logits / self.cfg.temperature
        return logits, labels

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x = torch.cat(x, dim=0)
        features = self.model(x, y)
        logits, labels = self.info_nce_loss(features)
        loss = self.criterion(logits, labels)
        self.train_acc(logits, labels)
        self.train_precision(logits, labels)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True)
        self.log('train_precision', self.train_precision, on_step=True, on_epoch=True)
        self.log("train_loss", loss)
        return loss















def _get_optimizer(name, params, lr, weight_decay, momentum):
    if name == "LAMB":
        optimizer = optim.Lamb(params, lr, weight_decay=weight_decay)
    elif name == "SGD":
        optimizer = torch.optim.SGD(params, lr, momentum=momentum, weight_decay=weight_decay)
    elif name == "Adam":
        optimizer = torch.optim.Adam(params, lr, weight_decay=weight_decay, eps=1e-4)
    elif name == "AdamW":
        optimizer = torch.optim.Adam(params, lr, weight_decay=weight_decay, eps=1e-4)
    else:
        raise Exception(f"Invalid optimizer name: ${name}")
    return optimizer


def _get_scheduler(name, optimizer, epochs, lr):
    if name == "StepLR":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
    elif name == "CosineAnnealingLR":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizereta_min=1e-5, last_epoch=-1)
    elif name == "ReduceLROnPlateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=1, threshold=0.001)
    elif name == "triangular":
        min_lr = 0.1 * lr
        max_lr = 1.0 * lr
        return torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=min_lr, max_lr=max_lr, step_size_up=4,
                                                 step_size_down=4, mode=name, last_epoch=-1,
                                                 cycle_momentum=False)
    else:
        return scheduler

class SuprEngine(ExperimentEngine):

    def __init__(self, model, cfg, epochs,class_dim=8000):
        super(SuprEngine, self).__init__(model=model, cfg=cfg, epochs=epochs)
        self.softmax = torch.nn.Softmax(dim=1)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.val_criterion = torch.nn.CrossEntropyLoss()
        self.val_acc = torchmetrics.Accuracy()
        self.val_precision = torchmetrics.Precision()
            
    def configure_optimizers(self):
        optimizer = _get_optimizer(name=self.cfg.optimizer,
                                   # params=filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.cfg.lr,
                                   params=self.model.parameters(), lr=self.cfg.lr,
                                   weight_decay=self.cfg.weight_decay, momentum=self.cfg.momentum)
        lr_scheduler = _get_scheduler(name=self.cfg.scheduler, optimizer=optimizer, epochs=self.epochs, lr=self.cfg.lr)        
        if lr_scheduler is None:
            return [optimizer]
        return [optimizer], [lr_scheduler]

    def training_step(self, train_batch, batch_idx):
        x, labels = train_batch
        logits = self.model(x, labels)
        loss = self.criterion(logits, labels)
        self.train_acc(logits, labels)
        self.train_precision(logits, labels)        
       
        # loss, logits, preds = self.model(x, labels)
        # self.train_acc(preds, labels)
        # self.train_precision(preds, labels)
        
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_precision', self.train_precision, on_step=True, on_epoch=True)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, labels = val_batch
        logits = self.model(x, labels)
        loss = self.val_criterion(logits, labels)
        self.val_acc(logits, labels)
        self.val_precision(logits, labels)

        # loss, logits, preds = self.model(x, labels)
        # self.val_acc(preds, labels)
        # self.val_precision(preds, labels)

        self.log('val_acc', self.val_acc, on_step=True, on_epoch=True)
        self.log('val_precision', self.val_precision, on_step=True, on_epoch=True)
        self.log("val_loss", loss, on_step=True, on_epoch=True)

        return loss

    def on_train_epoch_start(self):
        self.train_logits = torch.zeros(0, dtype=torch.long, device=self.device)
        self.train_labels = torch.zeros(0, dtype=torch.long, device=self.device)
