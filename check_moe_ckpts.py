import deepspeed
import torch
def compare_ckpt(ckpt_path1, ckpt_path2):
    ckpt1 = torch.load(ckpt_path1, map_location='cpu')
    ckpt2 = torch.load(ckpt_path2, map_location='cpu')
    for (k1,v1), (k2,v2) in zip(ckpt1.items(), ckpt2.items()):
        if torch.allclose(v1, v2):
            print(f'{k1} = {k2}')
        else:
            print(f'{k1} != {k2}')

prefix="s3://P_model_weights/zengzhiyuan/checkpoints/moe/moe_1b_32e_from_scratch/"

compare_ckpt(prefix+'global_step10000/layer_1_expert_3_mp_rank_00_model_states.pt', 
            prefix+'global_step10000/layer_1_expert_8_mp_rank_00_model_states.pt')
