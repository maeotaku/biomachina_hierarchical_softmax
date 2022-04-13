import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch_optimizer as optim
import torchmetrics

from metrics.mrr import MRR


class ExperimentEngine(pl.LightningModule):

    def __init__(self, model, loader, loader_val, cfg):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.loader = loader
        self.loader_val = loader_val
        self.train_acc = torchmetrics.Accuracy()
        self.train_precision = torchmetrics.Precision()
        self.batch_size = loader.batch_size  # needed for automatic batch size calculation

    def train_dataloader(self):
        return self.loader

    def forward(self, x):
        return self.model(x)

class SimCLREngine(ExperimentEngine):

    def __init__(self, model, loader, loader_val, cfg):
        super(SimCLREngine, self).__init__(model=model, loader=loader, loader_val=loader_val, cfg=cfg)
        self.criterion = torch.nn.CrossEntropyLoss()

    def configure_optimizers(self):
        optimizer = _get_optimizer(name=self.cfg.optimizer,
                                   params=filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.cfg.lr,
                                   weight_decay=self.cfg.weight_decay, momentum=self.cfg.momentum)
        scheduler = _get_scheduler(name=self.cfg.scheduler, optimizer=optimizer, loader=self.loader)

        return [optimizer], [scheduler]

    def info_nce_loss(self, features):
        labels = torch.cat([torch.arange(self.cfg.batch_size) for i in range(self.cfg.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

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
        x, _ = train_batch
        x = torch.cat(x, dim=0)
        features = self.model(x)
        logits, labels = self.info_nce_loss(features)
        loss = self.criterion(logits, labels)
        self.train_acc(logits, labels)
        self.train_precision(logits, labels)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True)
        self.log('train_precision', self.train_precision, on_step=True, on_epoch=True)
        self.log("train_loss", loss)
        return loss


class SuprEngine(ExperimentEngine):

    def __init__(self, model, loader, loader_val, cfg, class_dim):
        super(SuprEngine, self).__init__(model=model, loader=loader, loader_val=loader_val, cfg=cfg)
        self.class_dim = class_dim
        self.criterion = torch.nn.CrossEntropyLoss()
        self.val_criterion = torch.nn.CrossEntropyLoss()

        # self.train_balanced_acc = torchmetrics.Accuracy(num_classes=class_dim, average='weighted')
        # self.train_auroc = torchmetrics.AUROC(num_classes=class_dim, average='weighted')
        # self.train_average_precision = torchmetrics.AveragePrecision(num_classes=class_dim, average='weighted')
        # self.train_f1 = torchmetrics.F1Score(num_classes=class_dim, average='weighted')
        # self.train_cohen_kappa = torchmetrics.CohenKappa(num_classes=class_dim)
        # self.train_matthews = torchmetrics.MatthewsCorrCoef(num_classes=class_dim)

        self.train_mrr = MRR()
        self.val_mrr = MRR()

        if loader_val:
            self.val_acc = torchmetrics.Accuracy()
            self.val_precision = torchmetrics.Precision()

    def configure_optimizers(self):
        optimizer = _get_optimizer(name=self.cfg.optimizer,
                                   # params=filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.cfg.lr,
                                   params=self.model.parameters(), lr=self.cfg.lr,
                                   weight_decay=self.cfg.weight_decay, momentum=self.cfg.momentum)
        scheduler = _get_scheduler(name=self.cfg.scheduler, optimizer=optimizer, loader=self.loader)
        return [optimizer], [scheduler]

    def training_step(self, train_batch, batch_idx):
        x, labels = train_batch
        loss, logits, preds = self.model(x, labels)
        # logits = self.model(x)
        # loss = self.criterion(logits, labels)
        self.train_acc(logits, labels)
        self.train_precision(logits, labels)
        self.train_mrr(preds, labels)

        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_precision', self.train_precision, on_step=True, on_epoch=True)
        self.log('train_mrr', self.train_mrr, on_step=False, on_epoch=True, prog_bar=True)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, labels = val_batch
        loss, logits, preds = self.model(x, labels)
        # logits = self.model(x)
        # loss = self.criterion(logits, labels)
        self.val_acc(logits, labels)
        self.val_precision(logits, labels)
        self.val_mrr(preds, labels)
        self.log('val_acc', self.val_acc, on_step=True, on_epoch=True)
        self.log('val_precision', self.val_precision, on_step=True, on_epoch=True)
        self.log("val_loss", loss, on_step=True, on_epoch=True)
        self.log('val_mrr', self.val_mrr, on_step=False, on_epoch=True)

        return loss

    def on_train_epoch_start(self):
        self.train_logits = torch.zeros(0, dtype=torch.long, device='cuda')
        self.train_labels = torch.zeros(0, dtype=torch.long, device='cuda')

    # def on_train_epoch_end(self):
    #     self.train_balanced_acc(self.train_logits, self.train_labels)
    #     self.train_auroc(self.train_logits, self.train_labels)
    #     self.train_average_precision(self.train_logits, self.train_labels)
    #     self.train_f1(self.train_logits, self.train_labels)
    #     self.train_cohen_kappa(self.train_logits, self.train_labels)
    #     self.train_matthews(self.train_logits, self.train_labels)
    # self.log('train_balanced_acc', self.train_balanced_acc, on_step=False, on_epoch=True)
    # self.log('train_auroc', self.train_auroc, on_step=False, on_epoch=True)
    # self.log('train_average_precision', self.train_average_precision, on_step=False, on_epoch=True)
    # self.log('train_f1', self.train_f1, on_step=False, on_epoch=True)
    # self.log('train_cohen_kappa', self.train_cohen_kappa, on_step=False, on_epoch=True)
    # self.log('train_matthews', self.train_matthews, on_step=False, on_epoch=True)


def _get_optimizer(name, params, lr, weight_decay, momentum):
    if name == "LAMB":
        optimizer = optim.Lamb(params, lr, weight_decay=weight_decay)
    elif name == "SGD":
        optimizer = torch.optim.SGD(params, lr, momentum=momentum, weight_decay=weight_decay)
    elif name == "Adam":
        optimizer = torch.optim.Adam(params, lr, weight_decay=weight_decay, eps=1e-4)
    elif name == "AdamW":
        optimizer = torch.optim.Adam(params, lr, weight_decay=weight_decay, eps=1e-4)
    elif name == "SGD":
        optimizer = torch.optim.SGD(params, lr, momentum=momentum)
    else:
        raise Exception(f"Invalid optimizer name: ${name}")
    return optimizer


def _get_scheduler(name, optimizer, loader):
    if name == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(loader), eta_min=0,
                                                               last_epoch=-1)
    elif name == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    else:
        scheduler = None
    return scheduler
