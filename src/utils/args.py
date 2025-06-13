#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File        :   main.py
@CreateTime  :   2022/03/31 16:51:40
@Author      :   cuibingjian
@Version     :   1.0
@Contact     :   None
@Description :   None
'''

import os
import argparse

import torch

cur_dir = os.path.dirname(os.path.realpath(__file__))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None, type=str)
    parser.add_argument("--method", type=str, default="gplinker", help="method")
    parser.add_argument("--train_data_file", default=None, type=str)
    parser.add_argument("--eval_data_file", default=None, type=str)
    parser.add_argument("--test_data_file", default=None, type=str)
    parser.add_argument("--schema_file", default=None, type=str)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--model_name_or_path",  type=str, default="hfl/chinese-roberta-wwm-ext", help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--model_type", type=str, default="bert", help="Model type to use.", choices=["roformer", "bert", "chinesebert"])
    parser.add_argument("--tokenizer_name", type=str, default=None, help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--do_train", action="store_true", default=False, help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", default=False, help="Whether to run eval on the test set.")
    parser.add_argument("--do_predict", action="store_true", default=False, help="Whether to run eval on the test set.")
    parser.add_argument("--evaluate_during_training", action="store_true")
    parser.add_argument("--do_lower_case", action="store_true", default=True)

    parser.add_argument("--per_device_train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32, help="Batch size (per device) for the dev dataloader.")
    parser.add_argument("--per_device_test_batch_size", type=int, default=32, help="Batch size (per device) for the test dataloader.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="linear",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--num_warmup_steps_or_radios",
        type=eval,
        default=0.1,
        help="Number of steps or radios for the warmup in the lr scheduler.",
    )
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--weight_decay", type=float, default=0.02, help="Weight decay to use.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", type=int, default=20, help="Total number of training epochs to perform.")
    parser.add_argument("--max_train_steps", type=int, default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )

    parser.add_argument("--output_dir", type=str, default="./outputs/", help="Where to store the final model.")
    parser.add_argument("--cache_dir", type=str, default="data_caches", help="Where to store data caches.")
    parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory")

    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--fp16", action="store_true", help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    parser.add_argument("--logging_steps", type=int, default=200, help="logging_steps.")
    parser.add_argument("--save_steps", type=int, default=10804, help="save_steps.",)
    parser.add_argument("--gpu_ids", default=None)
    parser.add_argument('--adv_type', default=None, type=str, choices=['fgm', 'pgd', ''])
    parser.add_argument('--task_name', default=None, type=str)
    parser.add_argument("--doc_stride", default=None, type=int)
    parser.add_argument("--writer_type", default='tensorboard', type=str)
    parser.add_argument("--topk", type=int, default=1, help="save_topk.")

    args = parser.parse_args()

    # if not args.gpu_ids:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = ''
    # else:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    args.n_gpu = torch.cuda.device_count()
    args.device = torch.device("cuda" if torch.cuda.is_available() and args.gpu_ids else "cpu")
    return args
