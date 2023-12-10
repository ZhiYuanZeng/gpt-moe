import argparse
import os
from megatron.training import get_model, setup_optim, setup_model_and_optimizer
from megatron.checkpointing import load_checkpoint, save_checkpoint
from tools.ckpts.from_dense_to_moe import transform, check_copied_params, check_forward, dense_args_to_moe_args
from megatron.utils import print_rank_0
from megatron.initialize import initialize_megatron
import torch
import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
from megatron.neox_arguments import NeoXArgs
from megatron import mpu


def load_model(neox_args, use_cache=False):
    """Setup model and optimizer."""
    model = get_model(neox_args=neox_args, use_cache=use_cache)
    model, optimizer, lr_scheduler = setup_optim(model, neox_args) 
    if neox_args.load is not None:
        neox_args.iteration = load_checkpoint(
            neox_args=neox_args,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            iteration=neox_args.load_iteration,
        )
        print_rank_0(
            f"Loading checkpoint and starting from iteration {neox_args.iteration}"
        )
    return model

def main(args):
    neox_args = NeoXArgs.consume_neox_args(args)
    neox_args.configure_distributed_args()
    neox_args.build_tokenizer()  # tokenizer needs to be build in training in order to set the padding vocab

    neox_args.load = neox_args.from_dense_to_moe['dense_dir']

    # setup logging and timers
    initialize_megatron(neox_args=neox_args)

    torch.distributed.barrier()

    print_rank_0(f'loading dense model from {neox_args.load}')
    dense_model, _, _ = setup_model_and_optimizer(neox_args=neox_args, use_cache=True)
    
    neox_args = dense_args_to_moe_args(neox_args)
    neox_args.load = None

    print_rank_0('finish loading dense checkpoint')
    exit()
    moe_model, _, _ = setup_model_and_optimizer(neox_args=neox_args)

    moe_model = transform(dense_model=dense_model, moe_model=moe_model, moe_args=neox_args)
    
    neox_args.load = neox_args.from_dense_to_moe['moe_dir']
    neox_args.save = neox_args.from_dense_to_moe['moe_dir']

    save_checkpoint(
        neox_args=neox_args,
        iteration=0,
        model=moe_model,
        optimizer=None,
        lr_scheduler=None,
    )

    torch.distributed.barrier()

    # verify the conversion can be loaded
    moe_model = load_model(neox_args)
    
    print("==========================================")
    print("Converted checkpoint successfully loaded!")
    print("==========================================")

    check_copied_params(moe_model)
    # check_forward(dense_model, moe_model)