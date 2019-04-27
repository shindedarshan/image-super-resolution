import torch
import torch.nn as nn
from torch.nn import Module
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn import _reduction as _Reduction
from torch._jit_internal import weak_module, weak_script_method


class _Loss(Module):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(_Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction

@weak_module
class HingeEmbeddingLoss(_Loss):
    __constants__ = ['margin', 'reduction']

    def __init__(self, margin=1.0, size_average=None, reduce=None, reduction='mean'):
        super(HingeEmbeddingLoss, self).__init__(size_average, reduce, reduction)
        self.margin = margin

    @weak_script_method
    def forward(self, input, target):
        return F.hinge_embedding_loss(input, target, margin=self.margin, reduction=self.reduction)
    
@weak_module
class MSELoss(_Loss):
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(MSELoss, self).__init__(size_average, reduce, reduction)

    @weak_script_method
    def forward(self, input, target):
        return F.mse_loss(input, target, reduction=self.reduction)
    
class GANLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, D_label, target_label):
        g_loss = torch.zeros(1)
        batch_size = len(target_label)
        for i in range(len(target_label)):
          if target_label[i] == 1:
              g_loss += torch.min(torch.tensor([0, -1 + D_label[i].item()]))
          if target_label[i] == -1:
              g_loss += torch.min(torch.tensor([0, -1 - D_label[i].item()]))
        g_loss /= batch_size
        g_loss = Variable(g_loss, requires_grad = True)
        return g_loss