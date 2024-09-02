from typing import Any
import torch.nn as nn
import torch
import sys
from deepspeed.moe.layer import MoE, MOELayer, TopKGate
from deepspeed.moe.sharded_moe import einsum, _AllToAll, multiplicative_jitter, _capacity, _one_hot_to_float
from torch import Tensor
from typing import Any
import torch
from torch import Tensor
from torch.nn import Module


class ExpertChoiceMoE(MoE):
    def __init__(self, hidden_size, expert, num_experts=1, ep_size=1, k=1, capacity_factor=1, 
                 eval_capacity_factor=1, min_capacity=4, use_residual=False, noisy_gate_policy: str = None, 
                 drop_tokens: bool = True, use_rts=True, use_tutel: bool = False, enable_expert_tensor_parallelism: bool = False, 
                 aux_loss_weight: dict = None, use_elbo=False, experts=None, gate_st=False):
        super(ExpertChoiceMoE, self).__init__(
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
            gate_st=gate_st)
        self.deepspeed_moe = ExpertChoiceLayer.from_moe_layer(self.deepspeed_moe)

class ExpertChoiceLayer(MOELayer):
    @classmethod
    def from_moe_layer(cls, moe_layer:MOELayer):
        return cls(moe_layer.gate, moe_layer.experts, moe_layer.ep_group_name, moe_layer.ep_size, moe_layer.num_local_experts)

    def __init__(self,
                 gate: TopKGate,
                 experts: Module,
                 ep_group_name,
                 ep_size,
                 num_local_experts: int,
                 use_tutel: bool = False,
                 use_elbo = False) -> None:
        super(ExpertChoiceLayer, self).__init__(gate, experts, ep_group_name, ep_size, num_local_experts, use_tutel=use_tutel, use_elbo=use_elbo)
        self.gate = ExpertChoiceGate(gate.wg.weight.shape[1], 
                                     gate.wg.weight.shape[0], 
                                     gate.k, 
                                     gate.capacity_factor, 
                                     gate.eval_capacity_factor, 
                                     gate.min_capacity, 
                                     gate.noisy_gate_policy, 
                                     gate.drop_tokens, 
                                     gate.use_rts, 
                                     gate.aux_loss_weight, 
                                     gate.gate_st)
        self.shuffle=False

    def forward(self, *input: Tensor, **kwargs: Any) -> Tensor:
        if self.wall_clock_breakdown:
            self.timers('moe').start()

        # Implement Algorithm 2 from GShard paper.
        d_model = input[0].shape[-1]

        # Initial implementation -> Reshape into S tokens by dropping sequence dimension.
        # Reshape into G groups so that each group can distribute tokens equally
        # group_size = kwargs['group_size'] if 'group_size' in kwargs.keys() else 1
        reshaped_input = input[0].reshape(-1, d_model)
        
        self.l_aux, combine_weights, dispatch_mask, self.exp_counts = self.gate(
            reshaped_input, used_token=input[1])
        
        self.l_aux = torch.tensor(0.).type_as(reshaped_input)
        self.exp_counts = {}

        dispatched_input = einsum("ecs,sm->ecm", dispatch_mask.type_as(input[0]), reshaped_input)

        if self.wall_clock_breakdown:
            self.timers('falltoall').start()

        if self.ep_size != 1:
            dispatched_input = _AllToAll.apply(self.ep_group, dispatched_input)

        if self.wall_clock_breakdown:
            self.timers('falltoall').stop()
            self.time_falltoall = self.timers('falltoall').elapsed(reset=False)

        # Re-shape after all-to-all: ecm -> gecm
        dispatched_input = dispatched_input.reshape(self.ep_size, self.num_local_experts, -1, d_model)

        expert_output = self.experts(dispatched_input)

        if self.wall_clock_breakdown:
            self.timers('salltoall').start()
        
        if self.ep_size != 1:
            expert_output = _AllToAll.apply(self.ep_group, expert_output)

        if self.wall_clock_breakdown:
            self.timers('salltoall').stop()
            self.time_salltoall = self.timers('salltoall').elapsed(reset=False)

        # Re-shape back: gecm -> ecm
        expert_output = expert_output.reshape(self.ep_size * self.num_local_experts, -1, d_model)

        combined_output = einsum("ecs,ecm->sm", combine_weights.type_as(input[0]), expert_output)

        a = combined_output.reshape(input[0].shape)

        if self.wall_clock_breakdown:
            self.timers('moe').stop()
            self.time_moe = self.timers('moe').elapsed(reset=False)

        return a
    
class ExpertChoiceGate(TopKGate):
    def forward(self,
                input: torch.Tensor,
                used_token: torch.Tensor = None,
                use_tutel: bool = False,
                return_gates = False):  # type: ignore
        assert not self.training
        if self.wg.weight.dtype != torch.float32:
            self.wg = self.wg.float()
        input_fp32 = input.float()
        # input jittering
        if self.noisy_gate_policy == 'Jitter' and self.training:
            input_fp32 = multiplicative_jitter(input_fp32, device=input.device)
        logits = self.wg(input_fp32)
        
        if self.training:
            capacity_factor = self.capacity_factor
        else:
            capacity_factor = self.eval_capacity_factor
        capacity = _capacity(logits, torch.tensor(capacity_factor), torch.tensor(self.min_capacity))
        
        s, e, c = logits.shape[0], logits.shape[1], capacity.item()

        logits = logits.transpose(0,1)
        gates = torch.softmax(logits, dim=0) # e, s
        expert_capacity_indices = torch.topk(gates, dim=-1, k=capacity).indices # e,c
        assert expert_capacity_indices.shape == (e, c)
        
        dispatch_tensor = _one_hot_to_float(expert_capacity_indices, num_classes=torch.tensor(logits.shape[1])) # e,c,s
        assert dispatch_tensor.shape == (e, c, s)

        combine_tensor = dispatch_tensor * gates.unsqueeze(dim=1) # (e,c,s) x (e,1,s)
        assert combine_tensor.shape == (e, c, s)

        normalization = combine_tensor.sum(dim=0).sum(dim=0) # e,c,s -> c,s -> s
        normalization = normalization.unsqueeze(dim=0).unsqueeze(dim=0) # (1,1,s)
        assert normalization.shape == (1,1,s)

        combine_tensor = combine_tensor / (normalization+1e-6)

        return 0, dispatch_tensor, combine_tensor, None
