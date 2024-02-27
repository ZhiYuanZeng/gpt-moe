from typing import Any
import torch.nn as nn
import torch
import sys
from deepspeed.moe.layer import MoE, MOELayer, Experts
from torch import Tensor
from typing import Any
import torch
from torch import Tensor
from torch.nn import Module
import copy

class LocalExperts(torch.nn.Module):
    @classmethod
    def from_existing_experts(cls, experts: Experts, expert_group_name):
        return cls(expert_group_name=expert_group_name, existing_experts = experts.deepspeed_experts)

    def __init__(self, expert=None, num_local_experts=1, expert_group_name=None, existing_experts=None):
        super(LocalExperts, self).__init__()

        if expert is not None:
            self.deepspeed_experts = torch.nn.ModuleList([copy.deepcopy(expert) for i in range(num_local_experts)])
            self.num_local_experts = num_local_experts
        else:
            assert existing_experts is not None and isinstance(existing_experts, torch.nn.ModuleList)
            self.deepspeed_experts = existing_experts
            self.num_local_experts = len(self.deepspeed_experts)
        # TODO: revisit allreduce for moe.gate...
        for expert in self.deepspeed_experts:
            # TODO: Create param groups to handle expert + data case (e.g. param.group = moe_group)
            for name, param in expert.named_parameters():
                param.allreduce = False
                param.group_name = expert_group_name

    def _forward_with_chunks(self, inputs):
        chunks = inputs.chunk(self.num_local_experts, dim=0)
        expert_outputs = []
        for chunk, expert in zip(chunks, self.deepspeed_experts):
            out = expert(chunk)
            if type(out) is tuple:
                out = out[0]  # Ignore the bias term for now
            expert_outputs += [out]

        expert_output = torch.cat(expert_outputs, dim=0)
        return expert_output


    def forward(self, inputs, input_split=None):
        if input_split is None:
            return self._forward_with_chunks(inputs)
        
        assert isinstance(input_split, list)
        assert len(inputs.shape) == 2

        chunks = torch.split(inputs, split_size_or_sections = input_split, dim=0)
        assert len(chunks) == len(self.deepspeed_experts)
        
        expert_outputs = []
        skip_expert_outputs = 0.
        for i, (chunk, expert) in enumerate(zip(chunks, self.deepspeed_experts)):
            assert chunk.shape[0] == input_split[i]
            skip_expert = False
            try:
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



class BaseLayerMoE(MoE):
    def __init__(self, hidden_size, expert, num_experts=1, ep_size=1, k=1, capacity_factor=1, 
                 eval_capacity_factor=1, min_capacity=4, use_residual=False, noisy_gate_policy: str = None, 
                 drop_tokens: bool = True, use_rts=True, use_tutel: bool = False, enable_expert_tensor_parallelism: bool = False, 
                 aux_loss_weight: dict = None, use_elbo=False, experts=None, gate_st=False):
        super(BaseLayerMoE, self).__init__(
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
        self.deepspeed_moe = BaseLayer.from_moe_layer(self.deepspeed_moe)

class BaseLayer(MOELayer):
    @classmethod
    def from_moe_layer(cls, moe_layer:MOELayer):
        experts = LocalExperts.from_existing_experts(moe_layer.experts, expert_group_name=moe_layer.ep_group_name)
        return cls(moe_layer.gate, experts, moe_layer.ep_group_name, moe_layer.ep_size, moe_layer.num_local_experts)

    def __init__(self,
                 gate: Module,
                 experts: Module,
                 ep_group_name,
                 ep_size,
                 num_local_experts: int,
                 use_tutel: bool = False,
                 use_elbo = False) -> None:
        super(BaseLayer, self).__init__(gate, experts, ep_group_name, ep_size, num_local_experts, use_tutel=use_tutel, use_elbo=use_elbo)
        self.gate = BaseLayerGate(gate.wg, gate_st=gate.gate_st, ep_group=self.ep_group)
        self.shuffle=False

    def _set_ep_group(self, ep_group):
        self.ep_group = ep_group
        self.gate.ep_group = ep_group

    def forward(self, *input: Tensor, **kwargs: Any) -> Tensor:
        features = input[0].reshape(-1, input[0].size(-1))
        is_training = self.training

        if self.shuffle and is_training:
            # Send each token to a random worker, to break correlations within the batch
            shuffle_sort = torch.randperm(features.size(0), device=features.device)
            features = All2All.apply(features[shuffle_sort], self.ep_group)
        sort_by_expert, input_splits, output_splits, routing_probs = self.gate(features, is_training=True)
        # Swap these tokens for the right ones for our expert
        dispatched_input = features[sort_by_expert]
        routed_features = All2All.apply(
            dispatched_input, self.ep_group, input_splits, output_splits
        )
        
        if self.num_local_experts > 1:
            raise NotImplementedError(f"{self.num_local_experts=} is not supported")
        expert_outout = self.experts(routed_features, input_split=None)
            
        # Return to original worker and ordering
        expert_outout = All2All.apply(expert_outout, self.ep_group, output_splits, input_splits)
        recovered_outout = expert_outout[inverse_sort(sort_by_expert)] # (s,d)
        combined_outout = routing_probs.type_as(recovered_outout) * recovered_outout

        if self.shuffle and is_training:
            # Undo shuffling
            combined_outout = All2All.apply(combined_outout, self.ep_group)[inverse_sort(shuffle_sort)]
        self.l_aux = torch.tensor(0.).type_as(combined_outout)
        self.exp_counts = {}
        combined_outout = combined_outout.view_as(input[0])
        return combined_outout
    
def inverse_sort(order):
    # Creates an index that undoes a sort: xs==xs[order][inverse_sort(order)]
    return torch.empty_like(order).scatter_(
        0, order, torch.arange(0, order.size(0), device=order.device)
    )


class BaseLayerGate(nn.Module):
    def __init__(self, wg, gate_st, ep_group):
        super().__init__()
        self.num_workers = wg.weight.shape[0]
        self.wg = wg
        self.gate_st = gate_st
        self.cpp_balanced_assignment = self._load_assignment()
        self.ep_group = ep_group

    def forward(self, features, is_training, *args, **kwargs):
        if self.wg.weight.dtype != torch.float32:
            self.wg = self.wg.float()
        features = features.float()

        with torch.no_grad():
            # Compute similarity of each token to each expert, for routing
            token_expert_affinities = self.wg(features)

        # Compute which token goes to which expert
        if not is_training:
            raise NotImplementedError("the greedy assignment is under test")
        sort_by_expert, input_splits, output_splits = (
            self.balanced_assignment(token_expert_affinities)
            if is_training
            else self.greedy_assignment(token_expert_affinities)
        )
        routing_probs = torch.softmax(token_expert_affinities, dim=1)
        routing_probs = self.gather_scores(routing_probs, sort_by_expert, input_splits)
        if self.gate_st:
            routing_probs = routing_probs - routing_probs.detach() + 1
        
        return sort_by_expert, input_splits, output_splits, routing_probs
    
    def gather_scores(self, scores, sort_indices, input_splits):
        sorted_scores = scores[sort_indices]
        if input_splits is None:
            num_tokens_per_experts = scores.shape[0] // scores.shape[1]
            expert_indices = torch.tensor([i//num_tokens_per_experts for i in range(len(scores))])
        else:
            expert_indices = torch.tensor(
                [index for index, count in enumerate(input_splits) for _ in range(count)])
        expert_indices = expert_indices.unsqueeze(dim=-1).type_as(sort_indices)

        gathed_scores = torch.gather(input=sorted_scores, dim=1, index=expert_indices)
        i_sort = inverse_sort(sort_indices)
        return gathed_scores[i_sort]

    def balanced_assignment(self, scores):
        assert scores.shape[0] % scores.shape[1] == 0 
        ok = scores.isfinite()
        if not ok.all():
            # NaNs here can break the assignment algorithm
            scores[~ok] = scores[ok].min()
        
        input_split = [scores.shape[0] // scores.shape[1] for i in range(self.num_workers)]
        output_split = All2All.apply(torch.tensor(input_split, device=scores.device), self.ep_group).tolist()
        return self.cpp_balanced_assignment(scores, False), input_split, output_split
    
    # Assigns each token to the top k experts
    def greedy_assignment(self, scores, k=1):
        token_to_workers = torch.topk(scores, dim=1, k=k, largest=True).indices.view(-1)
        token_to_workers, sort_ordering = torch.sort(token_to_workers)
        worker2token = sort_ordering // k

        # Find how many tokens we're sending to each other worker (being careful for sending 0 tokens to some workers)
        input_splits = torch.zeros(
            (self.num_workers,), dtype=torch.long, device=scores.device
        )
        workers, counts = torch.unique_consecutive(token_to_workers, return_counts=True)
        input_splits[workers] = counts
        assert len(input_splits) == scores.shape[-1]
        assert input_splits.sum() == scores.shape[0]
        # Tell other workers how many tokens to expect from us
        output_splits = All2All.apply(input_splits, self.ep_group)
        return worker2token, input_splits.tolist(), output_splits.tolist()

    def _load_assignment(self):
        try:
            from balanced_assignment import balanced_assignment
            return balanced_assignment

        except ImportError as e:
            sys.stderr.write(
                "ERROR: missing balanced_assignment c++ module. run `python setup.py install`\n"
            )
            raise e

# Wraps torch.distributed.all_to_all_single as a function that supports autograd
class All2All(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xs, group=None, input_splits=None, output_splits=None):
        ctx.input_splits = input_splits
        ctx.output_splits = output_splits
        ctx.group = group
        ys = (
            torch.empty_like(xs)
            if output_splits is None
            else xs.new_empty(size=[sum(output_splits)] + list(xs.size()[1:]))
        )
        torch.distributed.all_to_all_single(
            ys, xs, output_split_sizes=output_splits, input_split_sizes=input_splits, group=group
        )
        return ys

    @staticmethod
    def backward(ctx, grad_output):
        result = (
            torch.empty_like(grad_output)
            if ctx.input_splits is None
            else grad_output.new_empty(
                size=[sum(ctx.input_splits)] + list(grad_output.size()[1:])
            )
        )
        torch.distributed.all_to_all_single(
            result,
            grad_output,
            output_split_sizes=ctx.input_splits,
            input_split_sizes=ctx.output_splits,
            group=ctx.group
        )
        return result, None, None, None