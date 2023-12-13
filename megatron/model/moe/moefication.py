from deepspeed.moe.layer import MoE, MOELayer, TopKGate, Experts
from deepspeed.moe.sharded_moe import (
    Any,
    Optional, 
    Tensor, 
    Tuple, 
)
from torch import Tensor
from torch.nn import Module
import torch
class MoeFromDense(MoE):
    """
    the moe-from-dense model should be identical with the dense model at the beginning of the training
    the top1 routing should not multiply probability on the expert outputs
    
    """
    def __init__(self, hidden_size, expert, num_experts=1, ep_size=1, k=1, capacity_factor=1, 
                 eval_capacity_factor=1, min_capacity=4, use_residual=False, noisy_gate_policy: str = None, 
                 drop_tokens: bool = True, use_rts=True, use_tutel: bool = False, enable_expert_tensor_parallelism: bool = False, 
                 aux_loss_weight: dict = None, use_elbo=False, experts=None, debug=False, **kwargs):
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
            experts=experts,
)
        
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
                                      use_elbo=use_elbo,
                                      debug=debug)
    

class MoeFromDenseDebug(MoeFromDense):
    def __init__(self, hidden_size, expert, num_experts=1, ep_size=1, k=1, capacity_factor=1, eval_capacity_factor=1, min_capacity=4, use_residual=False, noisy_gate_policy: str = None, drop_tokens: bool = True, use_rts=True, use_tutel: bool = False, enable_expert_tensor_parallelism: bool = False, aux_loss_weight: dict = None, use_elbo=False, experts=None, debug=False, **kwargs):
        drop_tokens = False
        super().__init__(hidden_size, expert, num_experts, ep_size, k, capacity_factor, eval_capacity_factor, 
                         min_capacity, use_residual, noisy_gate_policy, drop_tokens, use_rts, use_tutel, 
                         enable_expert_tensor_parallelism, aux_loss_weight, use_elbo, experts, debug=True, **kwargs)

class MoEFromDenseGate(TopKGate):
    def forward(self, input: Tensor, used_token: Tensor = None, use_tutel: bool = False, return_gates=False) -> Tuple[Tensor, Tensor, Tensor]:
        gate_output = super().forward(input, used_token, use_tutel, return_gates)
        l_aux, combine_weights, dispatch_mask, metadata = gate_output
        combine_weights = combine_weights-combine_weights.detach() + dispatch_mask
        return l_aux, combine_weights, dispatch_mask, metadata
    
class MoEFromDenseLayer(MOELayer):
    def __init__(self, gate: Module, experts: Module, ep_group_name, ep_size, num_local_experts: int, use_tutel: bool = False, use_elbo=False, debug=False) -> None:
        super().__init__(gate, experts, ep_group_name, ep_size, num_local_experts, use_tutel, use_elbo)
        self.debug = debug

    # def forward(self, *input: Tensor, **kwargs: Any) -> Tensor:
    #     if self.debug:
    #         self.l_aux = torch.tensor(0.).type_as(input[0])
    #         self.exp_counts = {}
    #         return self.experts.deepspeed_experts[1](input[0])[0]
    #     else:
    #         return super().forward(*input, **kwargs)