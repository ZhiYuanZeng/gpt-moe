import typing
from deepspeed.moe.layer import MoE, MOELayer, TopKGate, Experts
from deepspeed.moe.sharded_moe import (
    multiplicative_jitter, 
    Optional, 
    Tensor, 
    Tuple, 
    _capacity,
    _one_hot_to_float,
    einsum,
    _top_idx,
    AuxLoss,
    dist,
    math,
    F
)
import torch
from torch.distributions import Multinomial

class LayerAwareMoE(MoE):
    def __init__(self, hidden_size, expert, num_experts=1, ep_size=1, k=1, capacity_factor=1, 
                 eval_capacity_factor=1, min_capacity=4, use_residual=False, noisy_gate_policy: str = None, 
                 drop_tokens: bool = True, use_rts=True, use_tutel: bool = False, enable_expert_tensor_parallelism: bool = False, 
                 aux_loss_weight: dict = None, use_elbo=False, experts=None):
        super(LayerAwareMoE, self).__init__(
            hidden_size=hidden_size, 
            expert=expert, 
            num_experts=num_experts, 
            ep_size=ep_size, 
            k=k, 
            capacity_factor=capacity_factor, 
            eval_capacity_factor=eval_capacity_factor, 
            min_capacity=min_capacity, 
            use_residual=use_residual, 
            noisy_gate_policy=noisy_gate_policy, 
            drop_tokens=drop_tokens, 
            use_rts=use_rts, 
            use_tutel=use_tutel, 
            enable_expert_tensor_parallelism=enable_expert_tensor_parallelism, 
            aux_loss_weight=aux_loss_weight,
            use_elbo=use_elbo,
            experts=experts)
        
        experts = self.deepspeed_moe.experts # reused created experts to save memory
        GATE_CLS = LayerAwareGate
        self.deepspeed_moe = MOELayer(GATE_CLS(hidden_size, num_experts, k, capacity_factor, eval_capacity_factor,
                                               min_capacity, noisy_gate_policy, drop_tokens, use_rts, aux_loss_weight=aux_loss_weight,
                                               ),
                                      experts,
                                      self.expert_group_name,
                                      self.ep_size,
                                      self.num_local_experts,
                                      use_tutel=use_tutel,
                                      use_elbo=use_elbo)

    def set_layer_gate(self, num_layers, layer_idx, num_z, layer_gate_type):
        self.deepspeed_moe.gate.set_layer_gate(num_layers, layer_idx, num_z, layer_gate_type)

class LayerAwareGate(TopKGate):
    def set_layer_gate(self, num_layers, layer_idx, num_z, layer_gate_type):
        # p(e|x,l)=\sum p(z|l)p(e|x,z)
        assert num_layers != -1
        assert num_z != -1 
        assert layer_idx != -1
        self.num_experts = self.wg.weight.shape[0]
        self.layer_idx = layer_idx
        self.num_z = num_z
        assert self.num_experts % num_z == 0
        self.num_experts_per_z = self.num_experts//num_z

        assert layer_gate_type in ('product', 'add', 'none')
        if layer_gate_type == 'add':
            num_z = self.num_experts
        self.layer_logits = torch.nn.Parameter(torch.zeros(num_z))
        self.layer_gate_type = layer_gate_type
    
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
        layer_probs = torch.softmax(self.layer_logits, dim=-1) # (num_z,)
        
        if self.layer_gate_type == 'product':
            probs = torch.softmax(logits.view(logits.shape[0], self.num_z, self.num_experts_per_z), dim=-1)
            # (s,z,e/z) (z) -> (s,z,e/z) -> (s,e)
            probs = torch.einsum('szx,z->szx', probs, layer_probs).reshape(-1, self.num_experts)
            # assert torch.allclose(torch.sum(probs, dim=-1), torch.ones(probs.shape[0]).type_as(probs)), torch.sum(probs, dim=-1)[:10]
        elif self.layer_gate_type == 'add':
            # in fact it may not be necessary, since it is just a bias. the layer norm has bias.
            logits += self.layer_logits.unsqueeze(dim=0)
            probs = torch.softmax(logits, dim=-1) # (s,e)
        else:
            probs = torch.softmax(logits, dim=-1) # (s,e)

        if self.k == 1:
            l_aux, combine_weights, dispatch_mask, metadata = top1gating(probs, self.capacity_factor if self.training else self.eval_capacity_factor,
                                     self.min_capacity, used_token, self.noisy_gate_policy if self.training else None,
                                     self.drop_tokens, self.use_rts, use_tutel, aux_loss_weight=self.aux_loss_weight, return_gates=False)
            metadata[f'layer{self.layer_idx}_gates_histgram'] = Multinomial(self.num_experts, layer_probs).sample()
        else:
            raise NotImplementedError

        if self.wall_clock_breakdown:
            self.timers('TopKGate').stop()
            self.gate_time = self.timers('TopKGate').elapsed(reset=False)

        return l_aux, combine_weights, dispatch_mask, metadata

def top1gating(gates: Tensor,
               capacity_factor: float,
               min_capacity: int,
               used_token: Tensor = None,
               noisy_gate_policy: Optional[str] = None,
               drop_tokens: bool = True,
               use_rts: bool = True,
               use_tutel: bool = False,
               prioritized_routing: bool=True,
               aux_loss_weight: dict = None,
               return_gates: bool = False
               ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Implements Top1Gating on logits."""
    metadata = {}
    if noisy_gate_policy == 'RSample':
        raise NotImplementedError
        logits_w_noise = logits + gumbel_rsample(logits.shape, device=logits.device)
    # everything is in fp32 in this function
    original_gates = gates

    capacity = _capacity(gates, torch.tensor(capacity_factor), torch.tensor(min_capacity))

    # Create a mask for 1st's expert per token
    # noisy gating
    indices1_s = torch.argmax(gates, dim=1)
    num_experts = int(gates.shape[1])
    mask1 = F.one_hot(indices1_s, num_classes=num_experts)

    # logging
    SAMPLE_FRACTION = 0.2
    num_tokens, num_experts = gates.shape[0], gates.shape[1]
    expert1_hist = 100 * torch.histc((indices1_s.squeeze() + 1), bins=num_experts, min=1, max=num_experts) / num_tokens
    metadata["unused_expert1_count"] = (expert1_hist == 0).sum()
    expert1_hist = torch.sort(expert1_hist, dim=0, descending=True).values + torch.finfo(torch.float32).tiny

    sample_count = max(math.ceil(num_experts * SAMPLE_FRACTION), 1)
    metadata["expert1_balance_top"] = expert1_hist[:sample_count].sum()
    metadata["expert1_balance_bottom"] = expert1_hist[-sample_count:].sum()

    # mask only used tokens
    if used_token is not None:
        mask1 = einsum("s,se->se", used_token, mask1)

    # gating decisions
    exp_counts = torch.sum(mask1, dim=0).detach().to('cpu')

    # if we don't want to drop any tokens
    if not drop_tokens:
        new_capacity = torch.max(exp_counts).to(gates.device)
        dist.all_reduce(new_capacity, op=dist.ReduceOp.MAX, group=dist.get_world_group())
        capacity = new_capacity

    # Compute l_aux
    l_aux, loss_metadata = AuxLoss.get_auxloss(gates, mask1, aux_loss_weight['load_balance'], aux_loss_weight['zloss'], aux_loss_weight['entropy'])

    # Random Token Selection
    if use_rts:
        uniform = exp_selection_uniform_map.get(gates.device)
        if uniform is None:
            uniform = torch.distributions.uniform.Uniform(low=torch.tensor(0.0, device=gates.device),
                                                          high=torch.tensor(1.0, device=gates.device)).rsample
            exp_selection_uniform_map[gates.device] = uniform

        mask1_rand = mask1 * uniform(mask1.shape)
    else:
        mask1_rand = mask1

    assert gates.shape[
        0] >= min_capacity, "No. of tokens (batch-size) should be greater than min_capacity. Either set min_capacity to 0 or increase your batch size."

    top_idx = _top_idx(mask1_rand, capacity)

    new_mask1 = mask1 * torch.zeros_like(mask1).scatter_(0, top_idx, 1)
    mask1 = new_mask1

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
    
    if prioritized_routing:
        importance_scores = -1 * gates.max(dim=1)[0]
        sorted_mask1 = mask1[importance_scores.argsort(dim=0)]
        sorted_cumsum1 = (cumsum_fnc(sorted_mask1) - 1) * sorted_mask1
        importance_sorted_locations1 =  sorted_cumsum1[importance_scores.argsort(dim=0).argsort(dim=0)]
        locations1 = importance_sorted_locations1
    else:
        locations1 = cumsum_fnc(mask1) - 1

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
    locations1_s = torch.sum(locations1 * mask1, dim=1)

    # Normalize gate probabilities
    mask1_float = mask1.float()
    gates = gates * mask1_float

    locations1_sc = _one_hot_to_float(locations1_s, capacity)
    combine_weights = einsum("se,sc->sec", gates, locations1_sc)

    dispatch_mask = combine_weights.bool()

    is_untouted=~(dispatch_mask.any(-1).any(-1))
    metadata["unrouted_token_rate"]=is_untouted.sum()/is_untouted.size(0)
    metadata.update(loss_metadata)
    metadata['histgram1'] = expert1_hist # the histogram will be estimated in tensorboard
    if return_gates:
        return l_aux, combine_weights, dispatch_mask, metadata, original_gates
    else:
        return l_aux, combine_weights, dispatch_mask, metadata
