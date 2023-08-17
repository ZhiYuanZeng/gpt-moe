from megatron.model import GPT2ModelPipe
from megatron.model.transformer import ParallelLinearPipe, ParallelSelfAttention, ParallelMLP, ParallelTransformerLayer
from megatron.model.word_embeddings import EmbeddingPipe
from megatron import mpu
from megatron.model.norms import LayerNorm, RMSNorm, ScaleNorm
from megatron.model.moe_transformer import MoEParallelTransformerLayer
import torch.nn as nn
from copy import deepcopy
import torch
import torch.distributed as dist
from typing import List
from megatron.utils import print_rank_0

def iterate_submodules(model):
    all_modules=[]
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Module):
            all_modules.append(module)
            all_modules.extend(iterate_submodules(module))# Recursively iterate through submodules
    return all_modules

def copy_param(copy_from:List[nn.Module], copy_to:List[nn.Module]):
    for cf, ct in zip(copy_from, copy_to):
        for p1,p2 in zip(cf.parameters(), ct.parameters()):
            p2.data.copy_(p1.data)
            p2.is_from_dense = True

def copy_attention(copy_from:nn.Module, copy_to:nn.Module):
    attn_layers1 = [layer for layer in iterate_submodules(copy_from) if isinstance(layer, ParallelSelfAttention)]
    attn_layers2 = [layer for layer in iterate_submodules(copy_to) if isinstance(layer, ParallelSelfAttention)]
    assert len(attn_layers1) == len(attn_layers2)
    copy_param(attn_layers1, attn_layers2)

def copy_embedding(copy_from:nn.Module, copy_to:nn.Module):
    embed_layers1 = [layer for layer in iterate_submodules(copy_from) if isinstance(layer, EmbeddingPipe)]
    embed_layers2 = [layer for layer in iterate_submodules(copy_to) if isinstance(layer, EmbeddingPipe)]
    assert len(embed_layers1) == len(embed_layers2)
    copy_param(embed_layers1, embed_layers2)

def copy_final_linear(copy_from:nn.Module, copy_to:nn.Module):
    linear_layers1 = [layer for layer in iterate_submodules(copy_from) if isinstance(layer, ParallelLinearPipe)]
    linear_layers2 = [layer for layer in iterate_submodules(copy_to) if isinstance(layer, ParallelLinearPipe)]
    assert len(linear_layers1) == len(linear_layers2)
    copy_param(linear_layers1, linear_layers2)

def copy_layer_norm(copy_from:nn.Module, copy_to:nn.Module):
    norm_cls = (LayerNorm, RMSNorm, ScaleNorm)
    norm_layers1 = [layer for layer in iterate_submodules(copy_from) if isinstance(layer, norm_cls)]
    norm_layers2 = [layer for layer in iterate_submodules(copy_to) if isinstance(layer, norm_cls)]
    assert len(norm_layers1) == len(norm_layers2)
    copy_param(norm_layers1, norm_layers2)

def copy_shared_ffn(copy_from:nn.Module, copy_to:nn.Module):
    ffn_layers = [layer for layer in iterate_submodules(copy_from) if isinstance(layer, ParallelMLP)]
    shared_ffn_layers = [layer.mlp for layer in iterate_submodules(copy_to) if isinstance(layer, (MoEParallelTransformerLayer, ParallelTransformerLayer))]
    assert len(shared_ffn_layers) == len(ffn_layers)
    copy_param(ffn_layers, shared_ffn_layers)


def copy_expert_ffn(copy_from:nn.Module, copy_to:nn.Module, from_all=True):
    ffn_layers = [layer for layer in iterate_submodules(copy_from) if isinstance(layer, ParallelMLP)]
    ffn_layers = [layer for i, layer in enumerate(ffn_layers) if i%2==1]
    moe_ffn_layers = [layer.moe_layer.deepspeed_moe.experts for layer in iterate_submodules(copy_to) if isinstance(layer, MoEParallelTransformerLayer)]
    assert len(moe_ffn_layers) % len(ffn_layers) == 0, f'{len(ffn_layers)=}, {len(moe_ffn_layers)=}'
    num_layers_each_rank =len(moe_ffn_layers)//len(ffn_layers) # maybe num_local_experts > 1

    rank = dist.get_rank(group=None)
    world_size = dist.get_world_size(group=None)
    assert world_size * num_layers_each_rank == len(ffn_layers), f'{world_size=}, {num_layers_each_rank=}, {len(ffn_layers)=}, {len(moe_ffn_layers)=}'
    
    ffn_layers_of_current_rank = [i+rank*num_layers_each_rank for i in range(num_layers_each_rank)]
    for i,layer in enumerate(moe_ffn_layers):
        if from_all:
            ffn_layer_idx = ffn_layers_of_current_rank[i % num_layers_each_rank]
            print(f'{rank=}, {ffn_layers_of_current_rank=}, {ffn_layer_idx=}', flush=True)
        else:
            ffn_layer_idx = i // num_layers_each_rank
            print(f'{rank=}, {ffn_layer_idx=}')
        copy_param([ffn_layers[ffn_layer_idx],], [layer,])


def copy_shared_params(copy_from:nn.Module, copy_to:nn.Module):
    copy_attention(copy_from, copy_to)
    copy_shared_ffn(copy_from, copy_to)
    copy_layer_norm(copy_from, copy_to)
    copy_embedding(copy_from, copy_to)
    copy_final_linear(copy_from, copy_to)

def check_copied_params(model):
    from_dense_params = []
    random_initlized_params = []
    for n,p in model.named_parameters():
        if hasattr(p, 'is_from_dense') and p.is_from_dense:
            from_dense_params.append(n)
        else:
            random_initlized_params.append(n)
    print_rank_0(f'{from_dense_params=}')
    print_rank_0(f'{random_initlized_params=}')
    return from_dense_params, random_initlized_params

def zero_expert(model):
    moe_ffn_layers = [layer.moe_layer.deepspeed_moe.experts for layer in iterate_submodules(model) if isinstance(layer, MoEParallelTransformerLayer)]
    for module in moe_ffn_layers:
        for p in module.parameters():
            nn.init.zeros_(p)

def transform(dense_model, dense_args, build_model_func):
    expert_initialization = dense_args.expert_initialization
    assert expert_initialization in ("zero", "from_dense_single_layer", "from_dense_all_layers", "no"), expert_initialization
    assert dense_args.moe_freq == 0 # it is a dense model

    print(f"#######################expert_initialization: {expert_initialization}#######################")
    moe_args = dense_args # deepcopy fails
    if expert_initialization != "no":
        moe_args = set_moe_args(moe_args)
    moe_model = build_model_func(moe_args)
    copy_shared_params(dense_model, moe_model)

    if expert_initialization == 'from_dense_all_layers':
        copy_expert_ffn(dense_model, moe_model, from_all=True)
    elif expert_initialization == "from_dense_single_layer":
        copy_expert_ffn(dense_model, moe_model, from_all=False)
    elif expert_initialization == "zero":
        zero_expert(moe_model)    
    check_copied_params(moe_model)
    return moe_model

def set_moe_args(args):
    args.moe_freq = 2
    args.moe_num_experts = args.num_layers // 2
    args.moe_top_k = 1
    args.ep_world_size = torch.distributed.get_world_size(group=None)
    print_rank_0('changing the dense model setting to moe model setting by changing neo_args')
    return args