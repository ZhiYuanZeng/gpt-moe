# GPT-MoE
基于GPT-Neox和Deepspeed的MoE训练框架.

## Install
```
pip install -r requirements.txt
# 我们魔改了deepspeed的moe实现
pip install -e patched_deepspeed/

还需要安装一些CUDA kernel,参考GPT-Neox的README.md
```

## Features
- 支持MoE模型的Pretrain,以及基于Dense模型初始化的pretrain或者finetune
- 支持Top-k, Base Layer, Expert Choices, 和Recitify Router

## 缺陷
- 目前只测试了EP,暂时没测试过TP, PP, 和EP混合

## Training MoE
在训练模型之前需要将文件处理成.bin格式,具体参考GPT-Neox的README.md

训练需要调整两个文件,分别是启动shell文件,参考`tran_scripts/moe`,以及模型配置文件,参考`configs/moe/`

模型的config参数参考`megatron/neox_arguments/neox_args.py`

怎么基于一个Dense模型训练MoE模型? 当前的实现是Google提出的Upcycling,即复制Dense模型的参数,将Dense模型的FFN复制多份得到多个Experts. 参考train_scripts/moefication/convert_dense_to_moe.sh & train_scripts/moefication/convert_dense_to_moe.yml


## Citation
```
@misc{zeng2024turnwasteworthrectifying,
      title={Turn Waste into Worth: Rectifying Top-$k$ Router of MoE}, 
      author={Zhiyuan Zeng and Qipeng Guo and Zhaoye Fei and Zhangyue Yin and Yunhua Zhou and Linyang Li and Tianxiang Sun and Hang Yan and Dahua Lin and Xipeng Qiu},
      year={2024},
      eprint={2402.12399},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2402.12399}, 
}
```