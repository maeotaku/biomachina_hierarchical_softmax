import torch
import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, cohen_kappa_score, matthews_corrcoef

def get_mrr(predictions_dict):
    ranks = np.asarray([get_rank(value) for key, value in predictions_dict.items()])
    return np.sum((1 / ranks)) / len(predictions_dict)

def get_rank(dict_value):
    prob = dict_value['prob']
    label = dict_value['label']
    idx = np.argsort(prob)[::-1]
    np.argmax(prob) == label
    rank_i = np.squeeze(np.where(idx == label)) + 1

    return rank_i

def get_balanced_metrics(output, target):
    #Metrics for imbalanced datasets by sklearn
    pred = torch.argmax(outputs, dim=1)
    balanced_acc = balanced_accuracy_score(target, pred)
    f1 = f1_score(target, pred, average='weighted')
    cohen_kappa = cohen_kappa_score(target, pred)
    matthews = matthews_corrcoef(target, pred)
    return balanced_acc, f1, cohen_kappa, matthews