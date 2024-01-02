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
            assert getattr(p2, 'is_from_dense', False) == False # no duplicated copy
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

def iterate_ffns(model):
    modules = iterate_submodules(model)
    
    ffn_modules = []
    
    for module in modules:
        if isinstance(module, ParallelTransformerLayer):
            if isinstance(module, MoEParallelTransformerLayer):
                ffn_modules.append(module.experts.deepspeed_experts)
            else:
                ffn_modules.append(module.mlp)
    return ffn_modules    

def copy_ffn(copy_from:nn.Module, copy_to:nn.Module):
    source_ffn_modules, target_ffn_modules = iterate_ffns(copy_from), iterate_ffns(copy_to)
    assert len(source_ffn_modules) == len(target_ffn_modules)
    print_rank_0(f'there are {len(source_ffn_modules)} ffn modules to be shared')
    for s,t in zip(source_ffn_modules, target_ffn_modules):
        if isinstance(t, torch.nn.ModuleList): # experts
            for _t in t:
                copy_param([s,], [_t])
        else:
            copy_param([s,], [t,])

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
    with torch.no_grad():
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


def copy_dense_params_to_moe(dense_model, moe_model, moe_args, expert_init):
    print_rank_0('#'*20 + ' from dense to moe ' + '#'*20)
    print_rank_0(f'{expert_init=}')
    print_rank_0('#'*40)

    copy_attention(dense_model, moe_model)
    copy_layer_norm(dense_model, moe_model)
    copy_embedding(dense_model, moe_model)
    copy_final_linear(dense_model, moe_model)
    copy_ffn(dense_model, moe_model)

    return moe_model

def dense_args_to_moe_args(args):
    print_rank_0('#'*20 + ' from dense to moe '+'#'*20)
    moe_args = args.from_dense_to_moe
    for k,v in moe_args.items():
        if hasattr(args, k):
            print_rank_0(k, moe_args[k])
            setattr(args, k, moe_args[k])
        else:
            print_rank_0(f'{k} of from_dense_to_moe_args is not an attributed of neo_args')
    print_rank_0('#'*20 + ' from dense to moe '+'#'*20)
    return args