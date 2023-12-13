import argparse
import os
from megatron.training import get_model, setup_optim, setup_model_and_optimizer
from megatron.checkpointing import load_checkpoint, save_checkpoint
from tools.ckpts.from_dense_to_moe import copy_dense_params_to_moe, check_copied_params, check_forward, dense_args_to_moe_args
from megatron.utils import print_rank_0
from megatron.initialize import initialize_megatron
import torch
import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
from megatron.neox_arguments import NeoXArgs

def main(args):
    neox_args = NeoXArgs.consume_neox_args(args)
    neox_args.configure_distributed_args()
    neox_args.build_tokenizer()  # tokenizer needs to be build in training in order to set the padding vocab

    if 'dense_dir' in neox_args.from_dense_to_moe: 
        neox_args.load = neox_args.from_dense_to_moe['dense_dir']

    # setup logging and timers
    initialize_megatron(neox_args=neox_args)

    torch.distributed.barrier()

    print_rank_0(f'loading dense model from {neox_args.load}')
    dense_model, _, _ = setup_model_and_optimizer(neox_args=neox_args, use_cache=True)
    print_rank_0('finish loading dense checkpoint')
    
    # dense args to moe args
    neox_args = dense_args_to_moe_args(neox_args)
    neox_args.load = None

    # create moe model
    moe_model, _, _ = setup_model_and_optimizer(neox_args=neox_args, use_cache=True)
    moe_model = copy_dense_params_to_moe(dense_model=dense_model, moe_model=moe_model, moe_args=neox_args, expert_init= neox_args.from_dense_to_moe['expert_init'])
    
    if 'moe_dir' in neox_args.from_dense_to_moe: 
        neox_args.save = neox_args.from_dense_to_moe['moe_dir']
        save_checkpoint(
            neox_args=neox_args,
            iteration=0,
            model=moe_model,
            optimizer=None,
            lr_scheduler=None,
        )

    torch.distributed.barrier()
    
    print_rank_0("==========================================")
    print_rank_0("Converted checkpoint successfully loaded!")
    print_rank_0("==========================================")

    check_copied_params(moe_model)
    check_forward(dense_model, moe_model)