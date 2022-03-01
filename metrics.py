import torch
import numpy as np

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