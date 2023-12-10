from deepspeed.moe.layer import MoE, MOELayer, TopKGate, Experts
from deepspeed.moe.sharded_moe import (
    Any,
    multiplicative_jitter, 
    Optional, 
    Tensor, 
    Tuple, 
    exp_selection_uniform_map,
    _capacity,
    _one_hot_to_float,
    einsum,
    # tutel_moe,
    _top_idx,
    AuxLoss,
    dist,
    math,
    F
)
import torch
from torch import Tensor

class MoeFromDense(MoE):
    """
    the moe-from-dense model should be identical with the dense model at the beginning of the training
    the top1 routing should not multiply probability on the expert outputs
    
    """
    def __init__(self, hidden_size, expert, num_experts=1, ep_size=1, k=1, capacity_factor=1, 
                 eval_capacity_factor=1, min_capacity=4, use_residual=False, noisy_gate_policy: str = None, 
                 drop_tokens: bool = True, use_rts=True, use_tutel: bool = False, enable_expert_tensor_parallelism: bool = False, 
                 aux_loss_weight: dict = None, use_elbo=False, experts=None):
        super(MoeFromDense, self).__init__(
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
        GATE_CLS = MoEFromDenseGate
        self.deepspeed_moe = MoEFromDenseLayer(GATE_CLS(hidden_size, num_experts, k, capacity_factor, eval_capacity_factor,
                                               min_capacity, noisy_gate_policy, drop_tokens, use_rts, aux_loss_weight=aux_loss_weight,
                                               ),
                                      experts,
                                      self.expert_group_name,
                                      self.ep_size,
                                      self.num_local_experts,
                                      use_tutel=use_tutel,
                                      use_elbo=use_elbo)
    
    def set_mode(self, mode):
        assert mode in ('dense', 'moe')
        self.deepspeed_moe.set_mode(mode)

class MoEFromDenseGate(TopKGate):
    def forward(self, input: Tensor, used_token: Tensor = None, use_tutel: bool = False, return_gates=False) -> Tuple[Tensor, Tensor, Tensor]:
        gate_output = super().forward(input, used_token, use_tutel, return_gates)
        l_aux, combine_weights, dispatch_mask, metadata = gate_output
        combine_weights = combine_weights-combine_weights.detach() + dispatch_mask
        return l_aux, combine_weights, dispatch_mask, metadata
    
class MoEFromDenseLayer(MOELayer):
    def set_mode(self, mode):
        self.mode = mode

    def forward(self, *input: Tensor, **kwargs: Any) -> Tensor:
        if getattr(self, 'mode', 'moe') == 'dense':
            return self.experts[0](input[0])
        else:
            return super().forward(*input, **kwargs)