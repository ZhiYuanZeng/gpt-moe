import torch
from megatron import mpu, print_rank_0

class MoEStateManager:
    """
    may cause oom, don't forget to call get_xxx
    """
    def __init__(self) -> None:
        self.loss = []
        self.accum_loss = 0. # accum_loss is for grad accum

    def update_loss(self, loss):
        self.loss.append(loss)
    
    def get_loss(self, reduce_func='sum'):
        if len(self.loss)==0:
            return 0.
        
        if reduce_func=='sum':
            res = sum(self.loss)
        else:
            res = sum(self.loss)/len(self.loss)
        
        self.loss = []
        self.accum_loss += torch.tensor(res.item()).type_as(res)
        return res
    
    def get_accum_loss(self, reset=True):
        res = self.accum_loss
        if reset:
            self.accum_loss = 0.
        if not isinstance(res, torch.Tensor):
            return None
        return res

    def reset(self):
        self.loss = []
        self.accum_loss = 0.

moe_state_manager = MoEStateManager()

def cross_entropy(output, labels, _fp16=False):
    """From pretrain_gpt2:forward_step()"""
    """
    if self.fp16_lm_cross_entropy:
        assert output.dtype == torch.half
        loss = mpu.vocab_parallel_cross_entropy(output, labels)
    else:
        loss = mpu.vocab_parallel_cross_entropy(output.float(), labels)
        return loss
    """
    labels, loss_mask = labels[0], labels[1]
    if _fp16:
        assert output.dtype == torch.half and loss_mask.dtype == torch.half
        losses = mpu.vocab_parallel_cross_entropy(output.contiguous(), labels)
    else:
        losses = mpu.vocab_parallel_cross_entropy(output.float().contiguous(), labels)
    loss_mask = loss_mask.view(-1)
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum() # the mask is on padding, the loss is average over samples
    return loss

class CrossEntropy():
    def __init__(self, _fp16=False) -> None:
        self.fp16 = _fp16

    def forward(self, output, labels):
        return cross_entropy(output, labels, self.fp16)

class MoECrossEnropy():
    def __init__(self, moe_loss_weight=0.01, _fp16=False) -> None:
        self.moe_loss_weight=moe_loss_weight
        self.logging_loss = []
        self.fp16 = _fp16

    def forward(self, outputs, labels):
        assert isinstance(outputs, (list, tuple)) and len(outputs) == 2
        lm_outputs, moe_loss = outputs
        lm_loss = cross_entropy(lm_outputs, labels, self.fp16)
        self.logging_loss.append(self.moe_loss_weight * sum(moe_loss).detach())
        return self.moe_loss_weight * sum(moe_loss) + lm_loss # lm_loss is averaged on batch, moe_loss is also computed at batch-level

    def logging(self):
        if len(self.logging_loss) > 0:
            res = sum(self.logging_loss)/len(self.logging_loss)
            self.logging_loss = []
        else:
            res = 0.
        return res