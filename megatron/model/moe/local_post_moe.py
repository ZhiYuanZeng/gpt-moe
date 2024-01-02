import torch
from deepspeed.moe.sharded_moe import MOELayer, einsum, _AllToAll
from deepspeed.moe.layer import MoE, Experts
from torch.nn import Module

def assert_all_experts_are_same(experts):
    def assert_two_modules_are_same(m1, m2):
        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            assert p1.data.shape == p2.data.shape
            assert torch.allclose(p1.data, p2.data)
    for e in experts.experts:
        assert_two_modules_are_same(e, experts.experts[0])

def assert_close(x1, x2):
    assert torch.allclose(x1, x2, atol=1e-6), f'max distance:{torch.max(torch.abs(x1-x2))}'
    
class LocalPostMoE(MoE):
    def __init__(self, hidden_size, expert, num_experts=1, ep_size=1, k=1, capacity_factor=1, eval_capacity_factor=1, min_capacity=4, use_residual=False, noisy_gate_policy: str | None = None, drop_tokens: bool = True, use_rts=True, use_tutel: bool = False, enable_expert_tensor_parallelism: bool = False, aux_loss_weight: dict = None, use_elbo=False, experts=None, gate_st=False):
        super().__init__(hidden_size, expert, num_experts, ep_size, k, capacity_factor, eval_capacity_factor, min_capacity, use_residual, noisy_gate_policy, drop_tokens, use_rts, use_tutel, enable_expert_tensor_parallelism, aux_loss_weight, use_elbo, experts, gate_st)
        if experts is None:
            experts = Experts(expert, self.num_local_experts, self.expert_group_name)
        self.deepspeed_moe = LocalPostMoELayer.from_moe_layer(self.deepspeed_moe)

class LocalPostMoELayer(MOELayer):
    def __init__(self,
                    gate: Module,
                    experts: Module,
                    ep_group_name,
                    ep_size,
                    num_local_experts: int,
                    use_tutel: bool = False,
                    use_elbo = False) -> None:        
        assert_all_experts_are_same(experts)
        super().__init__(gate, experts, ep_group_name, ep_size, num_local_experts, use_tutel, use_elbo)

    @classmethod
    def from_moe_layer(cls, moe_layer:MOELayer):
        return cls(moe_layer.gate, moe_layer.experts, moe_layer.ep_group_name, moe_layer.ep_group, moe_layer.ep_size, moe_layer.num_local_experts)

    def forward(self, *inputs):
        # Implement Algorithm 2 from GShard paper.
        d_model = inputs[0].shape[-1]

        # Initial implementation -> Reshape into S tokens by dropping sequence dimension.
        # Reshape into G groups so that each group can distribute tokens equally
        # group_size = kwargs['group_size'] if 'group_size' in kwargs.keys() else 1
        reshaped_inputs = inputs[0].reshape(-1, d_model)

        self.l_aux, combine_weights, dispatch_mask, self.exp_counts = self.gate(reshaped_inputs, inputs[1])
        dispatched_inputs = einsum(
            "sec,sm->ecm", dispatch_mask.type_as(inputs[0]), reshaped_inputs
        )  # TODO: heavy memory usage due to long sequence length


        dispatched_inputs = _AllToAll.apply(self.ep_group, dispatched_inputs)

        # Re-shape after all-to-all: ecm -> gecm
        dispatched_inputs = dispatched_inputs.reshape(self.ep_size, self.num_local_experts, -1, d_model)

        expert_output = self.experts(dispatched_inputs)

        expert_output = _AllToAll.apply(self.ep_group, expert_output)

        # Re-shape back: gecm -> ecm
        expert_output = expert_output.reshape(self.ep_size * self.num_local_experts, -1, d_model)

        if self.gate.k == 1:
            combine_weights = combine_weights-combine_weights.detach() + dispatch_mask
        combined_output = einsum("sec,ecm->sm", combine_weights.type_as(inputs[0]), expert_output)

        routed_mask = (dispatch_mask!=0).any(-1).any(-1)
        
        dense_outputs = self.experts(reshaped_inputs)
        assert_close(combined_output[routed_mask], dense_outputs[routed_mask])
        combined_output[~routed_mask] = dense_outputs[~routed_mask] # if tokens are unrouted, computed at local devices
        
        out = combined_output.reshape(inputs[0].shape)
        
        return out