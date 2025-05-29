from torch.nn.modules.module import Module
from torch import Tensor
import torch


class SigmoidLoss(Module):
    def __init__(self):
        super(SigmoidLoss, self).__init__()
    
    def label_postprocess(self, target: Tensor, n_class: int) -> Tensor:
        assert len(target.shape) == 1, "only implemented for labels \in {0, ..., m}."
        multi_labels = -torch.ones((len(target), n_class))
        multi_labels[torch.arange(len(target)), target] = 1
        return multi_labels

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        target = self.label_postprocess(target, input.shape[1])
        loss = torch.sigmoid(-target * input).sum() / input.shape[0]
        return loss