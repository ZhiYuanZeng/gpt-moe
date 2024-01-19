from deepspeed.moe.sharded_moe import top1gating, dummy_top2gating, dummy_top1gating
import torch

logits = torch.randn(100,10)
logits.requires_grad_(True)
aux_loss_weight={
    'load_balance': 0.01,
    'zloss': 0.,
    'entropy': 0.
}

l_aux1, combine_weights1, dispatch_mask1, metadata1, gates1, location_sc1, capacity1  = dummy_top1gating(
    logits, capacity_factor=1, min_capacity=0, aux_loss_weight=aux_loss_weight, prioritized_routing=True)
loss1 = l_aux1 + combine_weights1.mean()
print(loss1)
loss1.backward()
grad1 = logits.grad

l_aux2, combine_weights2, dispatch_mask2, metadata2, gates2, location_sc2, capacity2 = dummy_top2gating(
    logits, capacity_factor=0.5, min_capacity=0, aux_loss_weight=aux_loss_weight, prioritized_routing=True)
loss2 = l_aux2 + combine_weights2.mean()
print(loss2)
loss2.backward()
grad2 = logits.grad

assert torch.all(dispatch_mask1 == dispatch_mask2)
assert l_aux1.item() == l_aux2.item()
assert torch.allclose(combine_weights1, combine_weights2)
assert torch.allclose(grad1, grad2)
print('check successfully!')