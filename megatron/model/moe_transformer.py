from megatron.model.transformer import ParallelTransformerLayer, ParallelLinear
from megatron.utils import print_rank_0
import deepspeed
import torch

class MoEParallelTransformerLayer(ParallelTransformerLayer):
    def __init__(self, neox_args, attention_mask_func, init_method, output_layer_init_method, layer_number, rpe=None, rotary=False, use_cache=False):
        super().__init__(neox_args, attention_mask_func, init_method, output_layer_init_method, layer_number, rpe, rotary, use_cache)
        self.moe_layer = deepspeed.moe.layer.MoE(
                        hidden_size=neox_args.hidden_size,
                        expert=self.mlp,
                        num_experts=neox_args.moe_num_experts,
                        ep_size=neox_args.ep_world_size,
                        use_residual=False,
                        k=neox_args.moe_top_k,
                        min_capacity=neox_args.moe_min_capacity,
                        noisy_gate_policy=neox_args.moe_noisy_gate_policy)
        for name,param in self.moe_layer.named_parameters():
            if 'bias' in name: # share bias paramters
                setattr(param, 'allreduce', True)
                delattr(param, 'group_name')
        self.adapter_gate_linear = torch.nn.Linear(neox_args.hidden_size, neox_args.hidden_size, bias=False)
        self.adapter_gate_activation = torch.nn.functional.silu
        torch.nn.init.zeros_(self.adapter_gate_linear.weight)

    def forward(self, x, attention_mask, all_l_auxs=[], layer_past=None):
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
            expert_output, l_aux, _ = self.moe_layer(x2)
            adapter_gate = self.adapter_gate_activation(self.adapter_gate_linear(x2))
            print_rank_0(f'gate value: {adapter_gate.mean()}')
            if self.mlp is not None and self.adapter_gate_linear is not None:
                shared_expert_output, mlp_bias = self.mlp(
                    x2
                )
                expert_output = adapter_gate * expert_output
                print_rank_0(f'{shared_expert_output.mean()=}, {expert_output.mean()=}')
                mlp_output = shared_expert_output + expert_output
            else:
                mlp_output = expert_output
                mlp_bias = self.mlp.dense_4h_to_h.bias

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
                attention_output = bias_dropout_fn(
                    attention_output,
                    bias=attention_bias.expand_as(residual),
                    residual=residual,
                    prob=self.hidden_dropout,
                )

            # output = x + mlp(ln2(x))
            attention_output = self.post_attention_layernorm(attention_output)
            expert_output, l_aux, _ = self.moe_layer(
                attention_output
            )
            adapter_gate = self.adapter_gate_activation(self.adapter_gate_linear(attention_output))
            if self.mlp is not None and self.adapter_gate_linear is not None:
                shared_expert_output, mlp_bias = self.mlp(
                    attention_output
                )
                mlp_output = shared_expert_output + adapter_gate * expert_output
            else:
                mlp_output = expert_output
                mlp_bias = self.mlp.dense_4h_to_h.bias

            with torch.enable_grad():
                output = bias_dropout_fn(
                    mlp_output,
                    bias=mlp_bias.expand_as(attention_output),
                    residual=attention_output,
                    prob=self.hidden_dropout,
                )
        all_l_auxs.append(l_aux) # record l_aux at each moe layer
        return output, all_l_auxs

class MoEParallelTransformerLayerPipe(MoEParallelTransformerLayer):
    """Extends ParallelTransformerLayer to forward attention_mask through the pipeline."""

    def forward(self, args):
        hidden_states, attention_mask = args[0], args[1]
        if len(args) == 2:
            all_l_auxs = []
        else:
            all_l_auxs = args[2]
        # we are returning just [hidden_states, mask]
        output, all_l_auxs = super().forward(hidden_states, attention_mask, all_l_auxs)
        return output, attention_mask, all_l_auxs

class MoEParallelLinearPipe(ParallelLinear):
    """Another helper class to pass presents through to the output when doing inference with a Pipe Parallel model"""

    def forward(self, args):
        assert len(args) == 2
        assert isinstance(args[0], torch.Tensor)
        assert isinstance(args[1], list)
        hidden_state = args
        logits, bias = super().forward(hidden_state)
        return logits, args[1]