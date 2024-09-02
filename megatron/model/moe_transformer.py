from megatron.model.transformer import ParallelTransformerLayer, ParallelLinear
from megatron import print_rank_0, print_rank
import deepspeed
import torch
import torch.distributed as dist
from megatron.model.moe.share_layer_moe import LayerAwareMoE
from megatron.model.moe.moefication import MoeFromDense
from megatron.model.moe.hier_moe import HierMoE
from megatron.model.moe.baselayer import BaseLayerMoE
from megatron.model.moe.expert_choices import ExpertChoiceMoE
from functools import partial 

class MoEParallelTransformerLayer(ParallelTransformerLayer):
    def __init__(self, neox_args, attention_mask_func, init_method, output_layer_init_method, layer_number, rpe=None, rotary=False, use_cache=False, experts = None):
        super().__init__(neox_args, attention_mask_func, init_method, output_layer_init_method, layer_number, rpe, rotary, use_cache)
        ep_world_size = min(neox_args.ep_world_size, torch.distributed.get_world_size())
        if neox_args.moe_share_layers is not None and neox_args.moe_share_layers['num_z']>1:
            MOE_CLS = LayerAwareMoE
        elif neox_args.from_dense_to_moe is not None:
            MOE_CLS = partial(MoeFromDense, **neox_args.from_dense_to_moe)
        elif neox_args.moe_base_layer:
            MOE_CLS = BaseLayerMoE
        elif neox_args.moe_expert_choices:
            MOE_CLS = ExpertChoiceMoE
        elif neox_args.hier_moe is not None:
            MOE_CLS = partial(HierMoE, **neox_args.hier_moe)
        else:
            MOE_CLS = deepspeed.moe.layer.MoE
        self.moe_layer = MOE_CLS(
                        hidden_size=neox_args.hidden_size,
                        expert=self.mlp,
                        num_experts=neox_args.moe_num_experts,
                        ep_size=ep_world_size,
                        k=neox_args.moe_top_k,
                        capacity_factor=neox_args.moe_capacity_factor,
                        eval_capacity_factor=neox_args.moe_eval_capacity_factor,
                        use_residual=neox_args.moe_use_residual,
                        min_capacity=neox_args.moe_min_capacity,
                        noisy_gate_policy=neox_args.moe_noisy_gate_policy,
                        aux_loss_weight=neox_args.moe_aux_loss_weight,
                        use_elbo=neox_args.moe_use_elbo,
                        experts=experts,
                        gate_st=neox_args.moe_gate_st,
                        drop_tokens=neox_args.moe_drop_tokens,
                        use_rts=neox_args.moe_use_rts)
        # assert neox_args.moe_aux_loss_weight is not None 
        # print_rank_0(neox_args.moe_aux_loss_weight)
        for name,param in self.moe_layer.named_parameters():
            if 'bias' in name: # share bias paramters
                setattr(param, 'allreduce', True)
                if hasattr(param, 'group_name'):
                    delattr(param, 'group_name')
        delattr(self, 'mlp')
        self.is_moe_layer = True

    @property
    def experts(self):
        return self.moe_layer.deepspeed_moe.experts

    @property
    def experts_weights(self):
        if self.mlp_type == 'llama':
            return [
                self.moe_layer.deepspeed_moe.experts.deepspeed_experts[0].w1.weight,
                self.moe_layer.deepspeed_moe.experts.deepspeed_experts[0].w2.weight,
                self.moe_layer.deepspeed_moe.experts.deepspeed_experts[0].w3.weight,
            ]
        else:
            return [
                self.moe_layer.deepspeed_moe.experts.deepspeed_experts[0].dense_4h_to_h.weight,
                self.moe_layer.deepspeed_moe.experts.deepspeed_experts[0].dense_h_to_4h.weight,
            ]
    
    def _check_same_num_elements(self, tensor):
        # Get the total number of elements in the tensor for comparison
        local_count = torch.tensor([tensor.numel()]).type_as(tensor)  # Get the number of elements in the local tensor

        # Gather the counts of elements from all processes
        all_counts = [torch.ones(1).type_as(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(all_counts, local_count)

        # Check if all counts are the same
        is_same = all(c.item() == local_count.item() for c in all_counts)

        assert is_same, "Number of elements in tensors across processes is not the same."

    def forward(self, x, attention_mask, all_l_auxs=None, all_metadata=None, layer_past=None):
        self._check_same_num_elements(x)
        layer_past = layer_past if layer_past is not None else self.layer_past
        bias_dropout_fn = self._get_bias_dropout()
        # x: [b, s, h]
        if self.gpt_j_residual:
            # pseudocode:
            # x = x + attn(ln(x)) + mlp(ln(x))
            # this means we can avoid doing the allreduce in the attn / mlp outputs
            # to save communication time (we can do a single allreduce after we add mlp / attn outputs).
            # due to a bug, the two layernorms are not tied in GPT-NeoX-20B. This is non-desirable, but
            # we preserve the functionality for backwards compatibility

            residual = x
            # applies the correct normalization depending on if the norms are tied
            if self.gpt_j_tied:
                x = self.input_layernorm(x)
                x1, x2 = x, x
            else:
                x1, x2 = self.input_layernorm(x), self.post_attention_layernorm(x)

            # attention operator
            attention_output, attention_bias = self.attention(
                x1, attention_mask, layer_past=layer_past
            )
            if self.use_cache:
                attention_output, presents = attention_output
                self.layer_past = presents

            with torch.enable_grad():
                attention_output = bias_dropout_fn(
                    attention_output,
                    bias=attention_bias.expand_as(attention_output),
                    residual=None,
                    prob=self.hidden_dropout,
                )

            # mlp operator
            # mlp_output, mlp_bias = self.mlp(x2)
            # l_aux = 0
            expert_output, l_aux, metadata = self.moe_layer(x2)
            mlp_output = expert_output
            mlp_bias = self.moe_layer.deepspeed_moe.experts.deepspeed_experts[0].dense_4h_to_h.bias
            with torch.enable_grad():
                output = bias_dropout_fn(
                    mlp_output,
                    bias=mlp_bias.expand_as(mlp_output),
                    residual=attention_output,
                    prob=self.hidden_dropout,
                )

            # output = (x + attn(ln(x)) + mlp(ln(x))
            output = residual + self.reduce(output)
        else:
            # pseudocode:
            # x = x + attn(ln1(x))
            # x = x + mlp(ln2(x))

            residual = x

            # x = x + attn(ln1(x))
            attention_output, attention_bias = self.attention(
                self.input_layernorm(x), attention_mask, layer_past=layer_past
            )

            if self.use_cache:
                attention_output, presents = attention_output
                self.layer_past = presents
            with torch.enable_grad():
                if attention_bias is not None:
                    # Use special bias_dropout_fn if we have a bias term from the above attention layer
                    attention_output = bias_dropout_fn(
                        attention_output,
                        bias=attention_bias.expand_as(residual),
                        residual=residual,
                        prob=self.hidden_dropout,
                    )
                else:
                    # Otherwise just apply dropout + residual
                    attention_output = (
                        torch.nn.functional.dropout(
                            attention_output,
                            p=self.hidden_dropout,
                            training=self.training,
                        )
                        + residual
                    )

            # output = x + mlp(ln2(x))
            expert_output, l_aux, metadata = self.moe_layer(
                self.post_attention_layernorm(attention_output)
            )
            mlp_output = expert_output
            with torch.enable_grad():
                if self.mlp_type == "llama":
                    # No dropout either
                    output = mlp_output + attention_output
                else:
                    mlp_bias = self.moe_layer.deepspeed_moe.experts.deepspeed_experts[0].dense_4h_to_h.bias
                    output = bias_dropout_fn(
                        mlp_output,
                        bias=mlp_bias.expand_as(attention_output),
                        residual=attention_output,
                        prob=self.hidden_dropout,
                    )
        if all_l_auxs is None:
            all_l_auxs = l_aux.unsqueeze(dim=-1)
        else:
            all_l_auxs=torch.cat([all_l_auxs, l_aux.unsqueeze(dim=-1)], dim=-1) # record l_aux at each moe layer
        if all_metadata is None:
            all_metadata = [metadata,]
        else:
            all_metadata.append(metadata)
        return output, all_l_auxs, all_metadata

class MoEParallelTransformerLayerPipe(MoEParallelTransformerLayer):
    """Extends ParallelTransformerLayer to forward attention_mask through the pipeline."""

    def forward(self, args):
        hidden_states, attention_mask = args[0], args[1]
        if len(args) == 2:
            all_l_auxs = None
            all_metadata = None
        else:
            all_l_auxs = args[2]
            all_metadata = args[3]
        # we are returning just [hidden_states, mask]
        output, all_l_auxs, all_metadata = super().forward(hidden_states, attention_mask, all_l_auxs, all_metadata)
        return output, attention_mask, all_l_auxs, all_metadata