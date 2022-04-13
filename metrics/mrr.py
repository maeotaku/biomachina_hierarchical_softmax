import torch
import torch.nn.functional as F


from torchmetrics import Metric

class MRR(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("rr", default=torch.tensor([]), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        ranks = self._mrr(preds, target, k=30)
        self.rr = torch.cat((self.rr, ranks), 0)
        return ranks.mean()

    def compute(self):
        return self.rr.mean()
    
    def _mrr(self, outputs: torch.Tensor, targets: torch.Tensor, k=100) -> torch.Tensor:
        """
        Source: ?
        """
        k = min(outputs.size(1), k)
        targets = F.one_hot(targets, num_classes=outputs.size(1))
        _, indices_for_sort = outputs.sort(descending=True, dim=-1)
        true_sorted_by_preds = torch.gather(
            targets, dim=-1, index=indices_for_sort
        )
        true_sorted_by_pred_shrink = true_sorted_by_preds[:, :k]

        values, indices = torch.max(true_sorted_by_pred_shrink, dim=1)
        indices = indices.type_as(values).unsqueeze(dim=0).t()
        result = torch.tensor(1.0) / (indices + torch.tensor(1.0))

        zero_sum_mask = values == 0.0
        result[zero_sum_mask] = 0.0
        return result
