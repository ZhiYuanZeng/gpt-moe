from megatron.model.transformer import ParallelLinearPipe, ParallelSelfAttention, ParallelMLP, ParallelTransformerLayer
from megatron.model.word_embeddings import EmbeddingPipe
from megatron.model.norms import LayerNorm, RMSNorm, ScaleNorm
from megatron.model.moe_transformer import MoEParallelTransformerLayer
import torch.nn as nn
import torch
import torch.distributed as dist
from typing import List
from megatron import print_rank_0, print_rank
from copy import deepcopy

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
            assert p1.data.shape == p2.data.shape
            print_rank_0(f'{p1.data.device=},{p2.data.device=}')
            p2.data = torch.from_numpy(p1.data.cpu().numpy()).type_as(p2.data)
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

def is_moe_layer(layer):
    return getattr(layer, 'is_moe_layer', False)

def iterate_ffn_layers(model):
    modules = iterate_submodules(model)

    ffn_modules = []
    
    for module in modules:
        if isinstance(module, ParallelTransformerLayer):
            ffn_modules.append(module)
    return ffn_modules

def copy_shared_ffn(copy_from:nn.Module, copy_to:nn.Module):
    source_ffn_modules, target_ffn_modules = iterate_ffn_layers(copy_from), iterate_ffn_layers(copy_to)
    nonmoe_source_ffn_modules = []
    nonmoe_target_ffn_modules = []
    for s_module, t_module in zip(source_ffn_modules, target_ffn_modules):
        if not is_moe_layer(t_module):
            nonmoe_source_ffn_modules.append(s_module.mlp)
            nonmoe_target_ffn_modules.append(t_module.mlp)

    copy_param(nonmoe_source_ffn_modules, nonmoe_target_ffn_modules)

def copy_expert_ffn(copy_from:nn.Module, copy_to:nn.Module):
    source_ffn_modules, target_ffn_modules = iterate_ffn_layers(copy_from), iterate_ffn_layers(copy_to)
    moe_source_ffn_modules = []
    moe_target_ffn_modules = []
    for s_module, t_module in zip(source_ffn_modules, target_ffn_modules):
        if is_moe_layer(t_module):
            moe_source_ffn_modules.append(s_module.mlp)
            moe_target_ffn_modules.append(t_module.experts.deepspeed_experts)
        
    assert len(moe_target_ffn_modules) <= len(moe_source_ffn_modules), f'{len(moe_source_ffn_modules)=}, {len(moe_target_ffn_modules)=}'
    if len(moe_source_ffn_modules) > len(moe_target_ffn_modules):
        moe_source_ffn_modules = moe_source_ffn_modules[-len(moe_target_ffn_modules):]
    for i in range(len(moe_target_ffn_modules)):
        for e in moe_target_ffn_modules[i]: # experts is a module list 
            copy_param([moe_source_ffn_modules[i],], [e,])

def copy_expert_ffn_wt_dropout(copy_from:nn.Module, copy_to:nn.Module, random_mask_rate:float):
    ffn_layers = [layer for layer in iterate_submodules(copy_from) if isinstance(layer, ParallelMLP)]
    ffn_layers = [layer for i, layer in enumerate(ffn_layers) if i%2==1]
    moe_ffn_layers = [layer.moe_layer.deepspeed_moe.experts for layer in iterate_submodules(copy_to) if isinstance(layer, MoEParallelTransformerLayer)]
    assert len(moe_ffn_layers) % len(ffn_layers) == 0, f'{len(ffn_layers)=}, {len(moe_ffn_layers)=}'
    num_layers_each_rank =len(moe_ffn_layers)//len(ffn_layers) # maybe num_local_experts > 1

    world_size = dist.get_world_size(group=None)
    assert world_size * num_layers_each_rank == len(ffn_layers), f'{world_size=}, {num_layers_each_rank=}, {len(ffn_layers)=}, {len(moe_ffn_layers)=}'
    
    for i,layer in enumerate(moe_ffn_layers):
        ffn_layer_idx = i // num_layers_each_rank
        src_param = ffn_layers[ffn_layer_idx]
        mask = torch.rand(src_param.size()) > random_mask_rate
        mask = mask.float()
        copy_param([src_param*mask,], [layer,])

def copy_expert_ffn_wt_pruning(copy_from:nn.Module, copy_to:nn.Module, ):
    pass

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
    # print_rank_0(f'{from_dense_params=}')
    print_rank_0('#'*20 + ' from dense to moe ' + '#'*20)
    print_rank_0(f'{random_initlized_params=}')
    print_rank_0('#'*40)
    return from_dense_params, random_initlized_params

def zero_expert(model):
    moe_ffn_layers = [layer.moe_layer.deepspeed_moe.experts for layer in iterate_submodules(model) if isinstance(layer, MoEParallelTransformerLayer)]
    for module in moe_ffn_layers:
        for p in module.parameters():
            nn.init.zeros_(p)

def check_forward(dense_model, moe_model):
    input_ids1 = torch.randint(size=(8,8), high=128, low=0)
    position_ids = torch.arange(0, 8)
    attention_mask = torch.full(size=(8,8), fill_value=False)
    
    param_device = next(dense_model.parameters()).device
    input_ids1, position_ids, attention_mask = input_ids1.to(param_device), position_ids.to(param_device), attention_mask.to(param_device)
    
    dense_outputs = dense_model((input_ids1, position_ids, attention_mask))
    moe_outputs = moe_model((input_ids1, position_ids, attention_mask))
    if isinstance(moe_outputs, tuple):
        moe_outputs = moe_outputs[0]
    assert dense_outputs.shape == moe_outputs.shape
    assert torch.allclose(dense_outputs, moe_outputs), f'max distance:{torch.max(torch.abs(dense_outputs - moe_outputs))}'
    print_rank_0('#'*10 + ' from dense to moe successful! ' + '#'*10)


def transform(dense_model, moe_model, moe_args, expert_init):
    print_rank_0('#'*20 + ' from dense to moe ' + '#'*20)
    print_rank_0(f'{expert_init=}')
    print_rank_0('#'*40)

    copy_shared_params(dense_model, moe_model) # attention, layernorm and half of ffns

    if expert_init == 'copy_and_share_moe_layers':
        raise NotImplementedError   
    elif expert_init == "copy":
        copy_expert_ffn(dense_model, moe_model)
    elif expert_init == "copy_and_mask":
        copy_expert_ffn_wt_dropout(dense_model, moe_model)   
    elif expert_init == "zero":
        zero_expert(moe_model)
    elif expert_init == "random":
        pass   
    else:
        raise NotImplementedError('only support expert_init of (copy, copy_and_share_moe_layers, copy_and_mask, zero, ramdom)')
    
    # check_copied_params(moe_model)

    # for layer in moe_model.sequential:
    #     if getattr(layer, 'is_moe_layer', False):
    #         layer.moe_layer.set_mode('dense')
    # check_forward(dense_model, moe_model) # make sure the params are copied successfully
    # for layer in moe_model.sequential:
    #     if getattr(layer, 'is_moe_layer', False):
    #         layer.moe_layer.set_mode('moe')

    return moe_model

def dense_args_to_moe_args(args):
    print_rank_0('#'*20 + ' from dense to moe '+'#'*20)
    moe_args = args.from_dense_to_moe
    for k,v in moe_args.items():
        if hasattr(args, k):
            print_rank(k, moe_args[k])
            setattr(args, k, moe_args[k])
        else:
            print_rank_0(f'{k} of from_dense_to_moe_args is not an attributed of neo_args')
    print_rank_0('#'*20 + ' from dense to moe '+'#'*20)
    return args