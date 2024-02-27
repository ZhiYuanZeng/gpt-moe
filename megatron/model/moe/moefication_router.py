from deepspeed.utils.timer import SynchronizedWallClockTimer
from deepspeed.utils import logger
from typing import Callable, Dict, TYPE_CHECKING, Any, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Module
import torch.nn.functional as F
from deepspeed.utils import groups
from deepspeed.moe.sharded_moe import drop_tokens, gather_tokens, einsum, gumbel_rsample, _capacity, dist, AuxLoss, _one_hot_to_float, TopKGate, multiplicative_jitter
import math


def top1gating_shift_priority(logits: Tensor,
               capacity_factor: float,
               min_capacity: int,
               used_token: Tensor = None,
               noisy_gate_policy: Optional[str] = None,
               drop_tokens: bool = True,
               use_rts: bool = True,
               use_tutel: bool = False,
               aux_loss_weight: dict = None,
               return_gates: bool = False,
               gate_st: bool = False,
               shift_priority = 0
               ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Implements Top1Gating on logits."""
    assert shift_priority != 0
    assert used_token is None
    assert aux_loss_weight['load_balance'] == 0.01

    metadata = {}
    if noisy_gate_policy == 'RSample':
        logits_w_noise = logits + gumbel_rsample(logits.shape, device=logits.device)
    # everything is in fp32 in this function
    gates = F.softmax(logits, dim=1)
    original_gates = gates

    capacity = _capacity(gates, torch.tensor(capacity_factor), torch.tensor(min_capacity))

    # Create a mask for 1st's expert per token
    # noisy gating
    indices1_s = torch.argmax(logits_w_noise if noisy_gate_policy == 'RSample' else gates, dim=1)
    num_experts = int(gates.shape[1])
    mask1 = F.one_hot(indices1_s, num_classes=num_experts)

    # mask only used tokens
    if used_token is not None:
        mask1 = einsum("s,se->se", used_token, mask1)

    # gating decisions
    exp_counts = torch.sum(mask1, dim=0).detach().to('cpu')

    # if we don't want to drop any tokens
    if not drop_tokens:
        new_capacity = torch.max(exp_counts).to(logits.device)
        dist.all_reduce(new_capacity, op=dist.ReduceOp.MAX, group=dist.get_world_group())
        logger.info(f'{new_capacity=}')
        capacity = new_capacity

    # Compute l_aux
    l_aux, loss_metadata = AuxLoss.get_auxloss(gates, mask1, aux_loss_weight['load_balance'], aux_loss_weight['zloss'], aux_loss_weight['entropy'])

    assert logits.shape[
        0] >= min_capacity, "No. of tokens (batch-size) should be greater than min_capacity. Either set min_capacity to 0 or increase your batch size."


    if use_tutel:
        # Tutel doesn't support index values masked with zero
        # so we need to replace masked indices with -1
        indices_mask = mask1.sum(dim=1) * num_experts - 1
        indices1_s = torch.min(indices1_s, indices_mask)

    # Compute locations in capacity buffer
    from functools import partial
    if use_tutel:
        cumsum_fnc = tutel_moe.fast_cumsum_sub_one
    else:
        cumsum_fnc = partial(torch.cumsum, dim=0)
    
    
    importance_scores = -1 * gates.max(dim=1)[0]
    sort_indices = importance_scores.argsort(dim=0)
    # shift
    shift_num = int(len(sort_indices) * shift_priority)
    sort_indices = torch.roll(sort_indices, shift_num)
    recover_indices = sort_indices.argsort(dim=0)
    
    sorted_mask1 = mask1[sort_indices]
    sorted_cumsum1 = (cumsum_fnc(sorted_mask1) - 1) * sorted_mask1
    importance_sorted_locations1 =  sorted_cumsum1[recover_indices]
    locations1 = importance_sorted_locations1

    if use_tutel:
        gates1_s = (gates * mask1).sum(dim=1)
        locations1_s = torch.sum(locations1 * mask1, dim=1)
        return l_aux, capacity, num_experts, [
            indices1_s,
        ], [
            locations1_s,
        ], [
            gates1_s,
        ], exp_counts

    # Store the capacity location for each token
    mask1 *= torch.lt(locations1, capacity)
    locations1_s = torch.sum(locations1 * mask1, dim=1)

    # Normalize gate probabilities
    mask1_float = mask1.float()
    gates = gates * mask1_float

    locations1_sc = _one_hot_to_float(locations1_s, capacity)
    combine_weights = einsum("se,sc->sec", gates, locations1_sc)

    dispatch_mask = combine_weights.bool()

    if gate_st:
        combine_weights = combine_weights-combine_weights.detach() + dispatch_mask

    is_untouted=~(dispatch_mask.any(-1).any(-1))
    metadata["unrouted_token_rate"]=is_untouted.sum()/is_untouted.size(0)
    metadata.update(loss_metadata)
    if return_gates:
        return l_aux, combine_weights, dispatch_mask, metadata, original_gates, gates.sum(dim=-1).detach()
    else:
        return l_aux, combine_weights, dispatch_mask, metadata

def top2gating_shift_priority(logits: Tensor, capacity_factor: float, min_capacity: int, prioritized_routing=True, aux_loss_weight=None, return_gates=False, shift_priority=0) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Implements Top2Gating on logits.
    Add
    - moe-batch-prioritized-routing
    - metadata
    """
    assert shift_priority != 0
    metadata={}
    # everything is in fp32 in this function
    gates = F.softmax(logits, dim=1)

    capacity = _capacity(gates, torch.tensor(capacity_factor * 2), torch.tensor(min_capacity))

    # Create a mask for 1st's expert per token
    indices1_s = torch.argmax(gates, dim=1)
    num_tokens = int(gates.shape[0])
    num_experts = int(gates.shape[1])
    mask1 = F.one_hot(indices1_s, num_classes=num_experts)

    # Create a mask for 2nd's expert per token using Gumbel-max trick
    # https://timvieira.github.io/blog/post/2014/07/31/gumbel-max-trick/
    if False:
        logits_w_noise = logits + gumbel_rsample(logits.shape, device=logits.device)
    else:
        logits_w_noise = logits
    # Replace top-expert with min value
    logits_except1 = logits_w_noise.masked_fill(mask1.bool(), float("-inf"))
    indices2_s = torch.argmax(logits_except1, dim=1)
    mask2 = F.one_hot(indices2_s, num_classes=num_experts)

    # Compute locations in capacity buffer
    if prioritized_routing:
        # sort tokens in the queue according to the gate values
        importance_scores = -1 * gates.max(dim=1)[0]
        sort_indices = importance_scores.argsort(dim=0)
        shift_num = int(len(sort_indices) * shift_priority)
        sort_indices = torch.roll(sort_indices, shift_num)
        recover_indices = sort_indices.argsort(dim=0)

        sorted_mask1 = mask1[sort_indices]
        sorted_cumsum1 = (torch.cumsum(sorted_mask1, dim=0) - 1) * sorted_mask1
        importance_sorted_locations1 =  sorted_cumsum1[recover_indices]

        sorted_mask2 = mask2[sort_indices]
        sorted_cumsum2 = (torch.cumsum(sorted_mask2, dim=0) - 1) * sorted_mask2
        importance_sorted_locations2 =  sorted_cumsum2[recover_indices]

        importance_sorted_locations2 += torch.sum(mask1, dim=0, keepdim=True)

        locations1, locations2 = importance_sorted_locations1, importance_sorted_locations2
    else:
        locations1 = torch.cumsum(mask1, dim=0) - 1
        locations2 = torch.cumsum(mask2, dim=0) - 1
        # Update 2nd's location by accounting for locations of 1st
        locations2 += torch.sum(mask1, dim=0, keepdim=True)

    # gating decisions
    exp_counts = torch.sum(mask1, dim=0).detach().to('cpu')

    # Compute l_aux
    l_aux, loss_metadata = AuxLoss.get_auxloss(gates, mask1, aux_loss_weight['load_balance'], aux_loss_weight['zloss'], aux_loss_weight['entropy'])

    metadata["overflow_expert1"] = 100 * torch.sum(mask1 * torch.ge(locations1, capacity)) / torch.sum(mask1)
    metadata["overflow_expert2"] = 100 * torch.sum(mask2 * torch.ge(locations2, capacity)) / torch.sum(mask2)

    # Remove locations outside capacity from mask
    mask1 *= torch.lt(locations1, capacity)
    mask2 *= torch.lt(locations2, capacity)

    # Store the capacity location for each token
    locations1_s = torch.sum(locations1 * mask1, dim=1)
    locations2_s = torch.sum(locations2 * mask2, dim=1)

    expert1_hist = 100 * torch.histc((indices1_s.squeeze() + 1), bins=num_experts, min=1, max=num_experts) / num_tokens
    metadata["unused_expert1_count"] = (expert1_hist == 0).sum()
    expert1_hist = torch.sort(expert1_hist, dim=0, descending=True).values + torch.finfo(torch.float32).tiny

    expert2_hist = 100 * torch.histc((indices2_s.squeeze() + 1), bins=num_experts, min=1, max=num_experts) / num_tokens
    metadata["unused_expert2_count"] = (expert2_hist == 0).sum()
    expert2_hist = torch.sort(expert2_hist, dim=0, descending=True).values +  torch.finfo(torch.float32).tiny

    SAMPLE_FRACTION = 0.2
    # expert1_balance_top: num tokens of top20% experts, num tokens of bottom20% experts,
    sample_count = max(math.ceil(num_experts * SAMPLE_FRACTION), 1)
    metadata["expert1_balance_top"] = expert1_hist[:sample_count].sum()
    metadata["expert1_balance_bottom"] = expert1_hist[-sample_count:].sum()
    metadata["expert2_balance_top"] = expert2_hist[:sample_count].sum()
    metadata["expert2_balance_bottom"] = expert2_hist[-sample_count:].sum()
    # metadata['histgram1'] = expert1_hist
    # metadata['histgram2'] = expert2_hist
    # Normalize gate probabilities
    mask1_float = mask1.float()
    mask2_float = mask2.float()
    gates1_s = einsum("se,se->s", gates, mask1_float)
    gates2_s = einsum("se,se->s", gates, mask2_float)
    denom_s = gates1_s + gates2_s
    # Avoid divide-by-zero
    denom_s = torch.clamp(denom_s, min=torch.finfo(denom_s.dtype).eps).detach()
    gates1_s /= denom_s
    gates2_s /= denom_s

    # Calculate combine_weights and dispatch_mask
    gates1 = einsum("s,se->se", gates1_s, mask1_float)
    gates2 = einsum("s,se->se", gates2_s, mask2_float)
    locations1_sc = _one_hot_to_float(locations1_s, capacity)
    locations2_sc = _one_hot_to_float(locations2_s, capacity)
    combine1_sec = einsum("se,sc->sec", gates1, locations1_sc)
    combine2_sec = einsum("se,sc->sec", gates2, locations2_sc)
    combine_weights = combine1_sec + combine2_sec
    dispatch_mask = combine_weights.bool()

    is_unrouted1 = ~((combine1_sec!=0).any(-1).any(-1))
    is_unrouted2 = ~((combine2_sec!=0).any(-1).any(-1))
    is_unrouted=~(dispatch_mask.any(-1).any(-1))
    metadata["unrouted_token_rate1"] = is_unrouted1.sum()/is_unrouted1.size(0)
    metadata["unrouted_token_rate2"] = is_unrouted2.sum()/is_unrouted2.size(0)
    metadata["unrouted_token_rate_all"]=is_unrouted.sum()/is_unrouted.size(0)
    for k,v in metadata.items():
        metadata[k] = v.detach()
    metadata.update(loss_metadata)
    if return_gates:
        return l_aux, combine_weights, dispatch_mask, metadata, gates, denom_s
    else:
        return l_aux, combine_weights, dispatch_mask, metadata

def first_max_second_kth_routing(logits: Tensor, capacity_factor: float, min_capacity: int, prioritized_routing=True, aux_loss_weight=None, return_gates=False, second_k=2) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Implements Top2Gating on logits.
    Add
    - moe-batch-prioritized-routing
    - metadata
    """
    metadata={}
    # everything is in fp32 in this function
    gates = F.softmax(logits, dim=1)
    capacity = _capacity(gates, torch.tensor(capacity_factor * 2), torch.tensor(min_capacity))

    # Create a mask for 1st's expert per token
    indices1_s = torch.argmax(gates, dim=1)
    num_tokens = int(gates.shape[0])
    num_experts = int(gates.shape[1])
    mask1 = F.one_hot(indices1_s, num_classes=num_experts)

    # Create a mask for 2nd's expert per token using Gumbel-max trick
    # Replace top-expert with min value
    indices2_s = torch.kthvalue(-logits, k=second_k, dim=-1).indices
    mask2 = F.one_hot(indices2_s, num_classes=num_experts)

    # Compute locations in capacity buffer
    if prioritized_routing:
        # sort tokens in the queue according to the gate values
        importance_scores = -1 * gates.max(dim=1)[0]
        sorted_mask1 = mask1[importance_scores.argsort(dim=0)]
        sorted_cumsum1 = (torch.cumsum(sorted_mask1, dim=0) - 1) * sorted_mask1
        importance_sorted_locations1 =  sorted_cumsum1[importance_scores.argsort(dim=0).argsort(dim=0)]

        sorted_mask2 = mask2[importance_scores.argsort(dim=0)]
        sorted_cumsum2 = (torch.cumsum(sorted_mask2, dim=0) - 1) * sorted_mask2
        importance_sorted_locations2 =  sorted_cumsum2[importance_scores.argsort(dim=0).argsort(dim=0)]

        importance_sorted_locations2 += torch.sum(mask1, dim=0, keepdim=True)

        locations1, locations2 = importance_sorted_locations1, importance_sorted_locations2
    else:
        locations1 = torch.cumsum(mask1, dim=0) - 1
        locations2 = torch.cumsum(mask2, dim=0) - 1
        # Update 2nd's location by accounting for locations of 1st
        locations2 += torch.sum(mask1, dim=0, keepdim=True)
    
    locations2 = torch.cumsum(mask2, dim=0) - 1
    # Update 2nd's location by accounting for locations of 1st
    locations2 += torch.sum(mask1, dim=0, keepdim=True)

    # gating decisions
    exp_counts = torch.sum(mask1, dim=0).detach().to('cpu')

    # Compute l_aux
    l_aux, loss_metadata = AuxLoss.get_auxloss(gates, mask1, aux_loss_weight['load_balance'], aux_loss_weight['zloss'], aux_loss_weight['entropy'])

    metadata["overflow_expert1"] = 100 * torch.sum(mask1 * torch.ge(locations1, capacity)) / torch.sum(mask1)
    metadata["overflow_expert2"] = 100 * torch.sum(mask2 * torch.ge(locations2, capacity)) / torch.sum(mask2)

    # Remove locations outside capacity from mask
    mask1 *= torch.lt(locations1, capacity)
    mask2 *= torch.lt(locations2, capacity)

    # Store the capacity location for each token
    locations1_s = torch.sum(locations1 * mask1, dim=1)
    locations2_s = torch.sum(locations2 * mask2, dim=1)

    expert1_hist = 100 * torch.histc((indices1_s.squeeze() + 1), bins=num_experts, min=1, max=num_experts) / num_tokens
    metadata["unused_expert1_count"] = (expert1_hist == 0).sum()
    expert1_hist = torch.sort(expert1_hist, dim=0, descending=True).values + torch.finfo(torch.float32).tiny

    expert2_hist = 100 * torch.histc((indices2_s.squeeze() + 1), bins=num_experts, min=1, max=num_experts) / num_tokens
    metadata["unused_expert2_count"] = (expert2_hist == 0).sum()
    expert2_hist = torch.sort(expert2_hist, dim=0, descending=True).values +  torch.finfo(torch.float32).tiny

    SAMPLE_FRACTION = 0.2
    # expert1_balance_top: num tokens of top20% experts, num tokens of bottom20% experts,
    sample_count = max(math.ceil(num_experts * SAMPLE_FRACTION), 1)
    metadata["expert1_balance_top"] = expert1_hist[:sample_count].sum()
    metadata["expert1_balance_bottom"] = expert1_hist[-sample_count:].sum()
    metadata["expert2_balance_top"] = expert2_hist[:sample_count].sum()
    metadata["expert2_balance_bottom"] = expert2_hist[-sample_count:].sum()
    # metadata['histgram1'] = expert1_hist
    # metadata['histgram2'] = expert2_hist
    # Normalize gate probabilities
    mask1_float = mask1.float()
    mask2_float = mask2.float()
    gates1_s = einsum("se,se->s", gates, mask1_float)
    gates2_s = einsum("se,se->s", gates, mask2_float)
    denom_s = gates1_s + gates2_s
    # Avoid divide-by-zero
    denom_s = torch.clamp(denom_s, min=torch.finfo(denom_s.dtype).eps).detach()
    gates1_s /= denom_s
    gates2_s /= denom_s

    # Calculate combine_weights and dispatch_mask
    gates1 = einsum("s,se->se", gates1_s, mask1_float)
    gates2 = einsum("s,se->se", gates2_s, mask2_float)
    locations1_sc = _one_hot_to_float(locations1_s, capacity)
    locations2_sc = _one_hot_to_float(locations2_s, capacity)
    combine1_sec = einsum("se,sc->sec", gates1, locations1_sc)
    combine2_sec = einsum("se,sc->sec", gates2, locations2_sc)
    combine_weights = combine1_sec + combine2_sec
    dispatch_mask = combine_weights.bool()

    is_unrouted1 = ~((combine1_sec!=0).any(-1).any(-1))
    is_unrouted2 = ~((combine2_sec!=0).any(-1).any(-1))
    is_unrouted=~(dispatch_mask.any(-1).any(-1))
    metadata["unrouted_token_rate1"] = is_unrouted1.sum()/is_unrouted1.size(0)
    metadata["unrouted_token_rate2"] = is_unrouted2.sum()/is_unrouted2.size(0)
    metadata["unrouted_token_rate_all"]=is_unrouted.sum()/is_unrouted.size(0)
    for k,v in metadata.items():
        metadata[k] = v.detach()
    metadata.update(loss_metadata)
    if return_gates:
        return l_aux, combine_weights, dispatch_mask, metadata, gates
    else:
        return l_aux, combine_weights, dispatch_mask, metadata

class ShiftPriorityTopKGate(Module):
    """Gate module which implements Top2Gating as described in Gshard_.
    ::

        gate = TopKGate(model_dim, num_experts)
        l_aux, combine_weights, dispatch_mask = gate(input)

    .. Gshard_: https://arxiv.org/pdf/2006.16668.pdf

    Args:
        model_dim (int):
            size of model embedding dimension
        num_experts (ints):
            number of experts in model
    """

    @classmethod
    def from_existing_gate(cls, existing_gate:TopKGate, shift_priority):
        assert shift_priority != 0
        gate = cls(
            wg = existing_gate.wg,
            k = existing_gate.k,
            capacity_factor = existing_gate.capacity_factor,
            eval_capacity_factor = existing_gate.eval_capacity_factor,
            min_capacity = existing_gate.min_capacity,
            noisy_gate_policy = existing_gate.noisy_gate_policy,
            drop_tokens = existing_gate.drop_tokens,
            use_rts = existing_gate.use_rts,
            aux_loss_weight = existing_gate.aux_loss_weight,
            gate_st = existing_gate.gate_st,
            shift_priority = shift_priority,
        )
        return gate

    def __init__(self,
                 wg: torch.nn.Linear = None,
                 k: int = 1,
                 capacity_factor: float = 1.0,
                 eval_capacity_factor: float = 1.0,
                 min_capacity: int = 8,
                 noisy_gate_policy: Optional[str] = None,
                 drop_tokens: bool = True,
                 use_rts: bool = True,
                 aux_loss_weight: dict = None,
                 gate_st:bool = False,
                 shift_priority = 0,
                 ) -> None:
        super().__init__()

        # Only top-1 and top-2 are supported at the moment.
        if k not in (1,2,3, -1, -2):
            raise ValueError('Only top-1, 2, 3, -1, -2 gatings are supported.')
        self.wg = wg
        self.k = k
        self.capacity_factor = capacity_factor
        self.eval_capacity_factor = eval_capacity_factor
        self.min_capacity = min_capacity
        self.noisy_gate_policy = noisy_gate_policy
        self.timers = SynchronizedWallClockTimer()
        self.wall_clock_breakdown = False
        self.gate_time = 0.0
        self.drop_tokens = drop_tokens
        self.use_rts = use_rts
        self.aux_loss_weight = aux_loss_weight
        self.gate_st = gate_st
        self.shift_priority = shift_priority

    def forward(self,
                input: torch.Tensor,
                used_token: torch.Tensor = None,
                use_tutel: bool = False,
                return_gates = False) -> Tuple[Tensor, Tensor, Tensor]:  # type: ignore

        if self.wall_clock_breakdown:
            self.timers('TopKGate').start()

        if self.wg.weight.dtype != torch.float32:
            self.wg = self.wg.float()
        input_fp32 = input.float()
        # input jittering
        if self.noisy_gate_policy == 'Jitter' and self.training:
            input_fp32 = multiplicative_jitter(input_fp32, device=input.device)
        logits = self.wg(input_fp32)

        if self.k == 1:
            gate_output = top1gating_shift_priority(logits, self.capacity_factor if self.training else self.eval_capacity_factor,
                                     self.min_capacity, used_token, self.noisy_gate_policy if self.training else None,
                                     self.drop_tokens, self.use_rts, use_tutel, aux_loss_weight=self.aux_loss_weight, 
                                     return_gates=return_gates, gate_st=self.gate_st, shift_priority=self.shift_priority)

        elif self.k == 2:
            gate_output = top2gating_shift_priority(logits, self.capacity_factor if self.training else self.eval_capacity_factor,
                                     self.min_capacity, aux_loss_weight=self.aux_loss_weight, return_gates=return_gates, shift_priority=self.shift_priority)
        else:
            raise NotImplementedError("k > 2 has not been supported currently")
        if self.wall_clock_breakdown:
            self.timers('TopKGate').stop()
            self.gate_time = self.timers('TopKGate').elapsed(reset=False)

        return gate_output
    
class KthGate(Module):
    """Gate module which implements Top2Gating as described in Gshard_.
    ::

        gate = TopKGate(model_dim, num_experts)
        l_aux, combine_weights, dispatch_mask = gate(input)

    .. Gshard_: https://arxiv.org/pdf/2006.16668.pdf

    Args:
        model_dim (int):
            size of model embedding dimension
        num_experts (ints):
            number of experts in model
    """

    @classmethod
    def from_existing_gate(cls, existing_gate:TopKGate, second_k:int):
        assert second_k != 2
        gate = cls(
            wg = existing_gate.wg,
            k = existing_gate.k,
            capacity_factor = existing_gate.capacity_factor,
            eval_capacity_factor = existing_gate.eval_capacity_factor,
            min_capacity = existing_gate.min_capacity,
            noisy_gate_policy = existing_gate.noisy_gate_policy,
            drop_tokens = existing_gate.drop_tokens,
            use_rts = existing_gate.use_rts,
            aux_loss_weight = existing_gate.aux_loss_weight,
            gate_st = existing_gate.gate_st,
            second_k = second_k,
        )
        return gate

    def __init__(self,
                 wg: torch.nn.Linear = None,
                 k: int = 1,
                 capacity_factor: float = 1.0,
                 eval_capacity_factor: float = 1.0,
                 min_capacity: int = 8,
                 noisy_gate_policy: Optional[str] = None,
                 drop_tokens: bool = True,
                 use_rts: bool = True,
                 aux_loss_weight: dict = None,
                 gate_st:bool = False,
                 second_k = 2,
                 ) -> None:
        super().__init__()

        # Only top-1 and top-2 are supported at the moment.
        if k not in (1,2,3, -1, -2):
            raise ValueError('Only top-1, 2, 3, -1, -2 gatings are supported.')
        self.wg = wg
        self.k = k
        self.capacity_factor = capacity_factor
        self.eval_capacity_factor = eval_capacity_factor
        self.min_capacity = min_capacity
        self.noisy_gate_policy = noisy_gate_policy
        self.timers = SynchronizedWallClockTimer()
        self.wall_clock_breakdown = False
        self.gate_time = 0.0
        self.drop_tokens = drop_tokens
        self.use_rts = use_rts
        self.aux_loss_weight = aux_loss_weight
        self.gate_st = gate_st
        self.second_k = second_k

    def forward(self,
                input: torch.Tensor,
                used_token: torch.Tensor = None,
                use_tutel: bool = False,
                return_gates = False) -> Tuple[Tensor, Tensor, Tensor]:  # type: ignore

        if self.wall_clock_breakdown:
            self.timers('TopKGate').start()

        if self.wg.weight.dtype != torch.float32:
            self.wg = self.wg.float()
        input_fp32 = input.float()
        logits = self.wg(input_fp32)

        assert self.k == 2
        gate_output = first_max_second_kth_routing(logits, self.capacity_factor if self.training else self.eval_capacity_factor,
                                    self.min_capacity, aux_loss_weight=self.aux_loss_weight, return_gates=return_gates, second_k=self.second_k)
        if self.wall_clock_breakdown:
            self.timers('TopKGate').stop()
            self.gate_time = self.timers('TopKGate').elapsed(reset=False)

        return gate_output
    