from deepspeed.moe.layer import MoE, MOELayer, TopKGate
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
    F,
    groups,
    _AllToAll,
    drop_tokens,
    gather_tokens
)
import torch
from torch import Tensor
from torch.distributions import Multinomial
from torch.nn import Module
import copy
from deepspeed.utils.logging import log_dist 

class HierMoE(MoE):
    """
    cross-gpu routing: normal top-k routing
    inside-gpu routing: routing without token dropout
    """
    def __init__(self, hidden_size, expert, num_experts=1, ep_size=1, k=1, capacity_factor=1, 
                 eval_capacity_factor=1, min_capacity=4, use_residual=False, noisy_gate_policy: str = None, 
                 drop_tokens: bool = True, use_rts=True, use_tutel: bool = False, enable_expert_tensor_parallelism: bool = False, 
                 aux_loss_weight: dict = None, use_elbo=False, experts=None, gate_st=False):
        super(HierMoE, self).__init__(
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
        
        if experts is None:
            experts = Experts(expert, self.num_local_experts, self.expert_group_name)
        crossgpu_gate = TopKGate(hidden_size, ep_size, k, capacity_factor, eval_capacity_factor, min_capacity, noisy_gate_policy, drop_tokens, \
                                  use_rts, aux_loss_weight=aux_loss_weight, gate_st=gate_st)
        insidegpu_gate = LocalGate(hidden_size, num_experts=self.num_local_experts, k=k, aux_loss_weight=aux_loss_weight, gate_st=gate_st)
        
        if ep_size != 1:
            self.deepspeed_moe = HierMoELayer(crossgpu_gate,
                                        insidegpu_gate,
                                        experts,
                                        self.expert_group_name,
                                        self.ep_size,
                                        self.num_local_experts,
                                        use_tutel=use_tutel,
                                        use_elbo=use_elbo)
        else:
            self.deepspeed_moe = LocalMoELayer(insidegpu_gate,
                                        experts,
                                        self.expert_group_name,
                                        self.ep_size,
                                        self.num_local_experts,
                                        use_tutel=use_tutel,
                                        use_elbo=use_elbo)

class Experts(torch.nn.Module):

    def __init__(self, expert, num_local_experts=1, expert_group_name=None):
        super(Experts, self).__init__()

        self.deepspeed_experts = torch.nn.ModuleList([copy.deepcopy(expert) for i in range(num_local_experts)])
        self.num_local_experts = num_local_experts
        
        # TODO: revisit allreduce for moe.gate...
        for expert in self.deepspeed_experts:
            # TODO: Create param groups to handle expert + data case (e.g. param.group = moe_group)
            for name, param in expert.named_parameters():
                param.allreduce = False
                param.group_name = expert_group_name

    def forward(self, inputs, input_split):
        assert isinstance(input_split, list)
        assert len(inputs.shape) == 2
        chunks = torch.split(inputs, split_size_or_sections = input_split, dim=0)
        assert len(chunks) == len(self.deepspeed_experts)
        expert_outputs = []
        skip_expert_outputs = 0.
        # log_dist(input_split)
        for chunk, expert in zip(chunks, self.deepspeed_experts):
            try:
                skip_expert = False
                if chunk.shape[0] == 0:
                    skip_expert = True
                    chunk = torch.zeros((1, inputs.shape[-1]), dtype=inputs.dtype, device=inputs.device)
                out = expert(chunk)

            except Exception:
                raise RuntimeError(f"the chunk size is : {chunk.shape}")
            if type(out) is tuple:
                out = out[0]  # Ignore the bias term for now
            if skip_expert:
                skip_expert_outputs += out.mean() * 0. # skipped outputs have zero grad, so the all-reduce will not be blocked
            else:
                expert_outputs += [out]
        assert skip_expert_outputs == 0 or skip_expert_outputs.item() == 0
        expert_output = torch.cat(expert_outputs, dim=0) + skip_expert_outputs
        assert expert_output.shape == inputs.shape
        return expert_output

class HierMoELayer(MOELayer):
    def __init__(self, crossgpu_gate: Module, insidegpu_gate: Module, experts: Module, ep_group_name, ep_size, num_local_experts: int, use_tutel: bool = False, use_elbo=False) -> None:
        super().__init__(crossgpu_gate, experts, ep_group_name, ep_size, num_local_experts=1, use_tutel=use_tutel, use_elbo=use_elbo)
        self.insidegpu_gate = insidegpu_gate

    def forward(self, *input: Tensor, **kwargs: Any) -> Tensor:
        if self.wall_clock_breakdown:
            self.timers('cross-moe').start()

        # Implement Algorithm 2 from GShard paper.
        d_model = input[0].shape[-1]

        # Initial implementation -> Reshape into S tokens by dropping sequence dimension.
        # Reshape into G groups so that each group can distribute tokens equally
        # group_size = kwargs['group_size'] if 'group_size' in kwargs.keys() else 1
        reshaped_input = input[0].reshape(-1, d_model)
        if self.use_elbo and self.training:
            shifted_input = torch.cat([input[0][1:,:], input[0][-1:,:]], dim=0) # input: (seq, bsz, d_model)
            reshaped_shifted_input = shifted_input.reshape(-1, d_model)
        
        if self.use_tutel:
            raise NotImplementedError
            _, C, E, indices_, locations_, gates_, self.exp_counts = self.gate(reshaped_input, input[1], True)
            S, M = reshaped_input.size(0), reshaped_input.size(1)

            if not hasattr(self, '_tutel_dispatcher'):
                self._tutel_dispatcher = tutel_moe.fast_dispatcher(E, C, M, dispatch_dtype=reshaped_input.dtype)

            assert not self.use_elbo
            self._tutel_dispatcher.update(indices_, locations_, gates_, capacity=C)
            dispatched_input = self._tutel_dispatcher.encode(reshaped_input)
        else:
            if self.use_elbo and self.training:
                self.l_aux, combine_weights, dispatch_mask, self.exp_counts= self.gate(
                    reshaped_input, shifted_input=reshaped_shifted_input, used_token=input[1])
            else:
                self.l_aux, combine_weights, dispatch_mask, self.exp_counts, crossgpu_probs = self.gate(
                    reshaped_input, used_token=input[1], return_gates=True)
                
            dispatched_input = einsum("sec,sm->ecm", dispatch_mask.type_as(input[0]), reshaped_input)
        if self.wall_clock_breakdown:
            self.timers('falltoall').start()

        if groups._get_expert_model_parallel_world_size() == 1:
            # If the non-expert is tensor-parallel, it will create
            # duplicate tokens on the tensor-parallel ranks.
            # Since our experts are not tensor-parallel, these duplicates
            # need to be dropped to ensure correctness.
            # this also doubles up as a communication optimization as we are
            # reducing the all-to-all communication volume.
            dispatched_input = drop_tokens(dispatched_input, dim=1)

        # TODO: if ep & tp, all2all can be improved to 2d, 
        # first split data in tp, and then all2all on ep, finally all-gather on tp
        if self.ep_size != 1:
            dispatched_input = _AllToAll.apply(self.ep_group, dispatched_input)

        if self.wall_clock_breakdown:
            self.timers('falltoall').stop()
            self.time_falltoall = self.timers('falltoall').elapsed(reset=False)

        # Re-shape after all-to-all: ecm -> gecm
        dispatched_input = dispatched_input.reshape(self.ep_size, self.num_local_experts, -1, d_model)
        expert_output, insidegpu_probs = self.local_moe(dispatched_input)
        assert dispatched_input.shape == expert_output.shape
        
        assert crossgpu_probs.shape[0] == insidegpu_probs.shape[0]
        global_probs = torch.einsum('bi,bj->bij', crossgpu_probs, insidegpu_probs).reshape(crossgpu_probs.shape[0], -1) 
        global_mask = F.one_hot(global_probs.argmax(dim=-1), num_classes=global_probs.shape[-1])
        aux_loss_weight = self.insidegpu_gate.aux_loss_weight
        self.l_aux, global_metadata =  AuxLoss.get_auxloss(global_probs, global_mask, aux_loss_weight['load_balance'], aux_loss_weight['zloss'], aux_loss_weight['entropy'])
        
        global_metadata = {'global_' + key: value for key, value in global_metadata.items()}
        self.exp_counts.update(global_metadata)

        if self.wall_clock_breakdown:
            self.timers('salltoall').start()
        
        if self.ep_size != 1:
            expert_output = _AllToAll.apply(self.ep_group, expert_output)

        if self.wall_clock_breakdown:
            self.timers('salltoall').stop()
            self.time_salltoall = self.timers('salltoall').elapsed(reset=False)

        # Re-shape back: gecm -> ecm
        expert_output = expert_output.reshape(self.ep_size * self.num_local_experts, -1, d_model)

        if groups._get_expert_model_parallel_world_size() == 1:
            # the dropped duplicate tokens need to be gathered on each
            # tensor parallel rank again for the tensor-parallel
            # non-expert of the next layer.
            expert_output = gather_tokens(expert_output, dim=1)

        if self.use_tutel:
            combined_output = self._tutel_dispatcher.decode(expert_output.view(E * C, M))
        else:
            combined_output = einsum("sec,ecm->sm", combine_weights.type_as(input[0]), expert_output)

        a = combined_output.reshape(input[0].shape)

        if self.wall_clock_breakdown:
            self.timers('cross-moe').stop()
            self.time_moe = self.timers('moe').elapsed(reset=False)

        return a
    
    def local_moe(self, inputs):
        if self.wall_clock_breakdown:
            self.timers('inside-moe').start()

        # Implement Algorithm 2 from GShard paper.
        input_shape = inputs.shape
        d_model = inputs.shape[-1]

        # Initial implementation -> Reshape into S tokens by dropping sequence dimension.
        # Reshape into G groups so that each group can distribute tokens equally
        # group_size = kwargs['group_size'] if 'group_size' in kwargs.keys() else 1
        reshaped_input = inputs.reshape(-1, d_model)
        sort_indices, reversed_ordering, combined_weights, input_splits, inside_probs = self.insidegpu_gate(reshaped_input)
        
        dispatched_input = reshaped_input[sort_indices]

        # expert_output = dispatched_input
        expert_output = self.experts(dispatched_input, input_splits)
        
        recovered_expert_output = expert_output[reversed_ordering]
        combined_output = recovered_expert_output * combined_weights.unsqueeze(dim=-1)
        if self.insidegpu_gate.k > 1:
            combined_output = combined_output.reshape(-1, self.insidegpu_gate.k, d_model).sum(dim=1)

        a = combined_output.reshape(input_shape)

        if self.wall_clock_breakdown:
            self.timers('inside-moe').stop()
            self.time_moe = self.timers('local_moe').elapsed(reset=False)

        return a, inside_probs

class LocalMoELayer(HierMoELayer):
    def __init__(self, insidegpu_gate: Module, experts: Module, ep_group_name, ep_size, num_local_experts: int, use_tutel: bool = False, use_elbo=False) -> None:
        super(HierMoELayer, self).__init__(insidegpu_gate, experts, ep_group_name, ep_size, num_local_experts, use_tutel=use_tutel, use_elbo=use_elbo)
        assert self.num_local_experts != 1
    
    @property
    def insidegpu_gate(self):
        return self.gate

    def forward(self, *input: Tensor, **kwargs: Any) -> Tensor:
        if self.wall_clock_breakdown:
            self.timers('local-moe').start()
        x = input[0]
        expert_output, probs = self.local_moe(x)
        mask = F.one_hot(probs.argmax(dim=-1), num_classes=probs.shape[-1])
        aux_loss_weight = self.gate.aux_loss_weight
        self.l_aux, self.exp_counts =  AuxLoss.get_auxloss(probs, mask, aux_loss_weight['load_balance'], aux_loss_weight['zloss'], aux_loss_weight['entropy'])
        
        if self.wall_clock_breakdown:
            self.timers('local-moe').stop()
            self.time_moe = self.timers('local-moe').elapsed(reset=False)
         
        return expert_output


def reverse_sort(order):
    # Creates an index that undoes a sort: xs==xs[order][inverse_sort(order)]
    return torch.empty_like(order).scatter_(
        0, order, torch.arange(0, order.size(0), device=order.device)
    )

class LocalGate(torch.nn.Module):
    def __init__(self,
                 model_dim: int,
                 num_experts: int,
                 aux_loss_weight: dict,
                 gate_st,
                 k: int = 1):
        super().__init__()
        self.wg = torch.nn.Linear(model_dim, num_experts, bias=False).float() 
        self.num_experts = num_experts
        self.k = k
        self.aux_loss_weight = aux_loss_weight
        self.gate_st = gate_st

    def forward(self,
                inputs: torch.Tensor) -> Tuple[Tensor, Tensor, list, Tensor, Tensor]:  # type: ignore
            logits = self.wg(inputs)
            probs = torch.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, dim=-1, k=self.k, largest=True)
            topk_indices = topk_indices.view(-1) # indices of expected experts for each token
            assert topk_indices.numel() == probs.shape[0] * self.k
            # mask = F.one_hot(probs.argmax(dim=-1), num_classes=self.num_experts)
            # assert mask.shape == probs.shape

            sorted_topk_indices, sort_ordering = torch.sort(topk_indices) # sort tokens according to the expert-id of their corresponding experts
            reversed_ordering = reverse_sort(sort_ordering)
            sort_ordering = sort_ordering // self.k # map token*k -> token

            # Find how many tokens we're sending to each other worker (being careful for sending 0 tokens to some workers)
            input_splits = torch.zeros(
                (self.num_experts,), dtype=torch.long, device=logits.device
            )
            workers, counts = torch.unique_consecutive(sorted_topk_indices, return_counts=True) # count how many tokens each expert received
            input_splits[workers] = counts

            if self.k > 1:
                combine_weights = torch.softmax(topk_probs, dim=-1)
            else:
                combine_weights = topk_probs
            combine_weights = combine_weights.view(-1).type_as(inputs)
            if self.gate_st:
                combine_weights = combine_weights-combine_weights.detach() + 1
            # aux_loss_weight = self.aux_loss_weight
            # l_aux, loss_metadata = AuxLoss.get_auxloss(probs, mask, aux_loss_weight['load_balance'], aux_loss_weight['zloss'], aux_loss_weight['entropy'])
            # loss_metadata = {'local_' + key: value for key, value in loss_metadata.items()}
            assert input_splits.sum() == len(sort_ordering)
            return sort_ordering, reversed_ordering, combine_weights, input_splits.tolist(), probs