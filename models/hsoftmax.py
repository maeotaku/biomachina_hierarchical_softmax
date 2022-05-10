import torch
import torch.nn as nn
import numpy as np

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

        # self.top = nn.Linear(self.nhid, self.nclasses)
        # self.bottom = nn.Linear(self.nclasses, self.nhid, self.ntokens_per_class)

        self.layer_top_W = nn.Parameter(torch.FloatTensor(self.nhid, self.nclasses), requires_grad=True)
        self.layer_top_b = nn.Parameter(torch.FloatTensor(self.nclasses), requires_grad=True)

        self.layer_bottom_W = nn.Parameter(torch.FloatTensor(self.nclasses, self.nhid, self.ntokens_per_class),
                                           requires_grad=True)
        self.layer_bottom_b = nn.Parameter(torch.FloatTensor(self.nclasses, self.ntokens_per_class), requires_grad=True)

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
        layer_top_probs = nn.functional.softmax(layer_top_logits)

        label_position_top = torch.argmax(layer_top_probs, dim=1)

        layer_bottom_logits = torch.squeeze(
            torch.bmm(torch.unsqueeze(inputs, dim=1), self.layer_bottom_W[label_position_top]), dim=1) + \
                              self.layer_bottom_b[label_position_top]
        layer_bottom_probs = nn.functional.softmax(layer_bottom_logits)

        return torch.bmm(layer_top_probs.unsqueeze(2), layer_bottom_probs.unsqueeze(1)).flatten(start_dim=1)

    def forward(self, inputs, labels=None):
        if labels is None:
            return self._predict(inputs)
        batch_size, d = inputs.size()

        layer_top_logits = torch.matmul(inputs, self.layer_top_W) + self.layer_top_b
        layer_top_probs = nn.functional.log_softmax(layer_top_logits)

        label_position_top = (labels / self.ntokens_per_class).long()
        label_position_bottom = (labels % self.ntokens_per_class).long()

        # print(layer_top_probs.shape, label_position_top.shape)

        layer_bottom_logits = torch.squeeze(
            torch.bmm(torch.unsqueeze(inputs, dim=1), self.layer_bottom_W[label_position_top]), dim=1) + \
                              self.layer_bottom_b[label_position_top]
        layer_bottom_probs = nn.functional.log_softmax(layer_bottom_logits)

        target_probs = layer_top_probs[torch.arange(batch_size).long(), label_position_top] + layer_bottom_probs[
            torch.arange(batch_size).long(), label_position_bottom]

        # print(f"top {layer_top_probs.shape} {layer_top_probs}")
        # print(f"bottom {layer_bottom_probs.shape} {layer_bottom_probs}")
        top_indx = torch.argmax(layer_top_probs, dim=1)
        botton_indx = torch.argmax(layer_bottom_probs, dim=1)

        real_indx = (top_indx * self.ntokens_per_class) + botton_indx
        # print(top_indx, self.nclasses, botton_indx)
        # print(f"target {target_probs.shape} {target_probs}")

#         lp = torch.add(layer_top_probs.unsqueeze(2), layer_bottom_probs.unsqueeze(1)).flatten(start_dim=1)
#         loss = torch.nn.functional.nll_loss(lp, labels)
#         loss = -torch.mean(torch.log(target_probs.type(torch.float32) + 1e-3))
#         loss = -torch.mean(torch.log(target_probs))
        # loss = torch.nn.functional.nll_loss(target_probs, labels)
        loss = -torch.mean(target_probs)
        with torch.no_grad():
            preds = self._predict(inputs)
            #torch.bmm(layer_top_probs.unsqueeze(2), layer_bottom_probs.unsqueeze(1)).flatten(start_dim=1)

        return loss, target_probs, layer_top_probs, layer_bottom_probs, top_indx, botton_indx, real_indx, preds

