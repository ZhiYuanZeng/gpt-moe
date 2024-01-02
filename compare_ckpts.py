import torch

ckpt_moe = torch.load('layer_0_expert_10_mp_rank_00_model_states.pt', map_location='cpu')
ckpt_dense = torch.load('neox/llama7B/global_step0/mp_rank_00_model_states.pt', map_location='cpu')

w_dense = ckpt_dense['module']['sequential.3.mlp.w1.weight']
w_moe=ckpt_moe['sequential.3.moe_layer.deepspeed_moe.experts.deepspeed_experts.10.w1.weight']


