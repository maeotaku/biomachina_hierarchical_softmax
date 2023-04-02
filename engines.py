import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch_optimizer as optim
import torchmetrics
# from torchmetrics import MetricCollection, ReciprocalRank, Accuracy, Precision, Recall, MulticlassAccuracy, MulticlassF1Score
import numpy as np

from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy, F1Score
from torcheval.metrics import ReciprocalRank

from metrics.mrr import MRR


class ExperimentEngine(pl.LightningModule):

    def __init__(self, model, cfg, epochs, class_dim):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.epochs = epochs

        metrics = MetricCollection(
            {
                "acc": torchmetrics.Accuracy(),
                "precision" : torchmetrics.Precision(),
                "f1score": torchmetrics.F1Score(),
            }
        )

        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")

    def forward(self, x):
        return self.model(x)

# class SimCLREngine(ExperimentEngine):

#     def __init__(self, model, loader, loader_val, cfg, epochs):
#         super(SimCLREngine, self).__init__(model=model, loader=loader, loader_val=loader_val, cfg=cfg, epochs=epochs)
#         self.criterion = torch.nn.CrossEntropyLoss()

#     def configure_optimizers(self):
#         optimizer = _get_optimizer(name=self.cfg.optimizer,
#                                    params=filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.cfg.lr,
#                                    weight_decay=self.cfg.weight_decay, momentum=self.cfg.momentum)
#         scheduler = _get_scheduler(name=self.cfg.scheduler, optimizer=optimizer, epochs=self.epochs)

#         return [optimizer], [scheduler]

#     def info_nce_loss(self, features):
#         labels = torch.cat([torch.arange(self.cfg.batch_size) for i in range(self.cfg.n_views)], dim=0)
#         labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
#         labels = labels.to(self.device)

#         features = F.normalize(features, dim=1)

#         similarity_matrix = torch.matmul(features, features.T)

#         # discard the main diagonal from both: labels and similarities matrix
#         mask = torch.eye(labels.shape[0], dtype=torch.bool)
#         labels = labels[~mask].view(labels.shape[0], -1)
#         similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
#         # assert similarity_matrix.shape == labels.shape

#         # select and combine multiple positives
#         positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

#         # select only the negatives
#         negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

#         logits = torch.cat([positives, negatives], dim=1)
#         labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

#         logits = logits / self.cfg.temperature
#         return logits, labels

#     def training_step(self, train_batch, batch_idx):
#         x, y = train_batch
#         x = torch.cat(x, dim=0)
#         features = self.model(x, y)
#         logits, labels = self.info_nce_loss(features)
#         loss = self.criterion(logits, labels)
#         self.train_acc(logits, labels)
#         self.train_precision(logits, labels)
#         self.log('train_acc', self.train_acc, on_step=True, on_epoch=True)
#         self.log('train_precision', self.train_precision, on_step=True, on_epoch=True)
#         self.log("train_loss", loss)
#         return loss















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

    def __init__(self, model, cfg, epochs, class_dim=8000):
        super(SuprEngine, self).__init__(model=model, cfg=cfg, epochs=epochs, class_dim=class_dim)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.val_criterion = torch.nn.CrossEntropyLoss()
            
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

        # logits = logits.argmax(dim=-1)
        self.train_metrics.update(logits, labels)
   
       
        # loss, logits, preds = self.model(x, labels)
        # self.train_acc(preds, labels)
        
        self.log_dict(self.train_metrics.compute(), on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, labels = val_batch
        logits = self.model(x, labels)
        loss = self.val_criterion(logits, labels)

        # logits = logits.argmax(dim=-1)
        self.val_metrics.update(logits, labels)

        # loss, logits, preds = self.model(x, labels)
        # self.val_acc(preds, labels)
        self.log_dict(self.val_metrics.compute(), on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss



class CombinedSuprEngine(ExperimentEngine):

    def __init__(self, model, datamodule, cfg, epochs, class_dim=8000):
        super(CombinedSuprEngine, self).__init__(model=model, cfg=cfg, epochs=epochs, class_dim=class_dim)
        self.datamodule = datamodule
        self.datamodule.fungi_indexes = self.datamodule.fungi_indexes.to(self.device)
        self.datamodule.plant_indexes = self.datamodule.plant_indexes.to(self.device)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.val_criterion = torch.nn.CrossEntropyLoss()

        metrics = MetricCollection(
            {
                "Accuracy": torchmetrics.Accuracy(),
                "Precision" : torchmetrics.Precision(),
                "F1Score": torchmetrics.F1Score(),
            }
        )

        self.fungi_train_metrics = metrics.clone(prefix="fungi_train_")
        self.fungi_val_metrics = metrics.clone(prefix="fungi_val_")
        self.plant_train_metrics = metrics.clone(prefix="plant_train_")
        self.plant_val_metrics = metrics.clone(prefix="plant_val_")
            
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
        

        with torch.no_grad():        

            self.train_metrics.update(logits, labels)
            self.log_dict(self.train_metrics.compute(), on_step=True, on_epoch=True, prog_bar=True)

            fungi_labels, fungi_logits = self._split_logits_labels(self.datamodule.fungi_indexes, labels, logits)
            if fungi_labels.shape[0] > 0:
                self.fungi_train_metrics.update(fungi_logits, fungi_labels)
                self.log_dict(self.fungi_train_metrics.compute(), on_step=True, on_epoch=True, prog_bar=True,  batch_size=fungi_labels.shape[0], rank_zero_only=True)
            else:
                self.fungi_train_metrics.update(torch.tensor([0]).to(self.device), torch.tensor([1]).to(self.device))
                self.log_dict(self.fungi_train_metrics.compute(), on_step=True, on_epoch=True, prog_bar=True,  batch_size=0, rank_zero_only=True)

            
            plant_labels, plant_logits = self._split_logits_labels(self.datamodule.plant_indexes, labels, logits)
            if plant_labels.shape[0] > 0:
                self.plant_train_metrics.update(plant_logits, plant_labels)
                self.log_dict(self.plant_train_metrics.compute(), on_step=True, on_epoch=True, prog_bar=True,  batch_size=plant_labels.shape[0], rank_zero_only=True)
            else:
                self.fungi_train_metrics.update(torch.tensor([0]).to(self.device), torch.tensor([1]).to(self.device))    
                self.log_dict(self.plant_train_metrics.compute(), on_step=True, on_epoch=True, prog_bar=True,  batch_size=0, rank_zero_only=True)
            
            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    # def training_epoch_end(self, outputs):
    #     self.log_dict(self.train_metrics.compute(), on_step=False, on_epoch=True)
    #     self.log_dict(self.fungi_train_metrics.compute(), on_step=False, on_epoch=True)
    #     self.log_dict(self.plant_train_metrics.compute(), on_step=False, on_epoch=True)
    #     self.train_metrics.reset()
    #     self.fungi_train_metrics.reset()
    #     self.plant_train_metrics.reset()


    
    def _split_logits_labels(self, indexes, labels, logits):
        idx = torch.nonzero(indexes[..., None].to(self.device) == labels.to(self.device))[:,1]
        return labels[idx], logits[idx,:]

    def validation_step(self, val_batch, batch_idx):
        x, labels = val_batch
        logits = self.model(x, labels)
        loss = self.val_criterion(logits, labels)

        with torch.no_grad():        

            self.val_metrics.update(logits, labels)
            self.log_dict(self.val_metrics.compute(), on_step=True, on_epoch=True, prog_bar=True)

            fungi_labels, fungi_logits = self._split_logits_labels(self.datamodule.fungi_indexes, labels, logits)
            if fungi_labels.shape[0] > 0:
                self.fungi_val_metrics.update(fungi_logits, fungi_labels)
                self.log_dict(self.fungi_val_metrics.compute(), on_step=True, on_epoch=True, prog_bar=True,  batch_size=fungi_labels.shape[0])
            else:
                self.fungi_val_metrics.update(torch.tensor([0]).to(self.device), torch.tensor([1]).to(self.device))
                self.log_dict(self.fungi_val_metrics.compute(), on_step=True, on_epoch=True, prog_bar=True,  batch_size=0)
            
            
            plant_labels, plant_logits = self._split_logits_labels(self.datamodule.plant_indexes, labels, logits)
            if plant_labels.shape[0] > 0:
                self.plant_val_metrics.update(plant_logits, plant_labels)
                self.log_dict(self.plant_val_metrics.compute(), on_step=True, on_epoch=True, prog_bar=True, batch_size=plant_labels.shape[0])
            else:
                self.plant_val_metrics.update(torch.tensor([0]).to(self.device), torch.tensor([1]).to(self.device))
                self.log_dict(self.plant_val_metrics.compute(), on_step=True, on_epoch=True, prog_bar=True, batch_size=0)
            
            self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)



        return loss
