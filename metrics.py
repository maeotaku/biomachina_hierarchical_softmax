import torch
import numpy as np

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

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