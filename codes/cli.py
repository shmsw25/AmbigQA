# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse
import logging

import random
import numpy as np
import torch

from run import run

def main():
    parser = argparse.ArgumentParser()

    ## Basic parameters
    parser.add_argument("--task",
                        default="qa-wo-context",
                        type=str)
    parser.add_argument("--train_file", default="data/nqopen-train.json",
                        type=str)
    parser.add_argument("--predict_file", default="data/nqopen-dev.json",
                        type=str)
    parser.add_argument("--output_dir", default=None, type=str, required=True)
    parser.add_argument("--dpr_data_dir", default="/checkpoint/sewonmin/dpr", type=str)
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_predict", action='store_true')
    parser.add_argument("--ambigqa", action='store_true')
    parser.add_argument("--skip_inference", action='store_true')
    parser.add_argument("--skip_db_load", action='store_true')
    parser.add_argument("--db_index", default=-1, type=int)
    parser.add_argument("--wiki_2020", action='store_true')

    ## Model parameters
    parser.add_argument('--bert_name', type=str, default='bert-base-uncased')
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--checkpoint", type=str,
                        help="Initial checkpoint (usually from a pre-trained BERT model).", \
                        default=None)
    parser.add_argument("--do_lowercase", action='store_true', default=True)

    # Preprocessing-related parameters
    parser.add_argument('--max_passage_length', type=int, default=200)
    parser.add_argument('--max_question_length', type=int, default=32)
    parser.add_argument('--train_M', type=int, default=24)
    parser.add_argument('--test_M', type=int, default=50)
    parser.add_argument("--append_another_bos", action='store_true')
    parser.add_argument('--n_jobs', type=int, default=12)
    parser.add_argument("--discard_not_found_answers", action='store_true')
    parser.add_argument("--psg_sel_dir", type=str, default=None)

    # Training-related parameters
    parser.add_argument("--train_batch_size", default=40, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--predict_batch_size", default=400, type=int,
                        help="Batch size per GPU/CPU for evaluation.")

    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--warmup_proportion", default=0.01, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=10000.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--wait_step', type=int, default=10)

    ## My learning parameters
    parser.add_argument("--max_n_answers", default=10, type=int)
    parser.add_argument("--loss_type", default="mml", type=str,
                        help="Learning method for weak supervision setting")
    parser.add_argument('--tau', type=float, default=12000.0)
    parser.add_argument("--dependent_learning", default="none", type=str,
                        help="none|{span|psg|both}-{soft|hard}]")

    ## Evaluation-related parameters
    parser.add_argument("--n_best_size", default=1, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json output file.")
    parser.add_argument("--max_answer_length", default=10, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")
    parser.add_argument("--verbose", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument('--eval_period', type=int, default=400,
                        help="Evaluate & save model")
    parser.add_argument('--prefix', type=str, default=None,
                        help="Prefix for saving predictions")
    parser.add_argument('--n_paragraphs', type=str, default=None)
    parser.add_argument("--save_psg_sel_only", action='store_true')
    parser.add_argument('--topk_answer', type=int, default=1)

    ## Other parameters
    parser.add_argument('--debug', action='store_true',
                        help="Use a subset of data for debugging")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        print("Output directory () already exists and is not empty.")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    ##### Start writing logs

    log_filename = "{}log.txt".format("" if args.do_train else "eval_")

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO,
                    handlers=[logging.FileHandler(os.path.join(args.output_dir, log_filename)),
                              logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    logger.info(args)
    logger.info(args.output_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.n_gpu = torch.cuda.device_count()

    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_predict:
        raise ValueError("At least one of `do_train` or `do_predict` must be True.")

    if args.do_train:
        if not args.train_file:
            raise ValueError("If `do_train` is True, then `train_file` must be specified.")
        if not args.predict_file:
            raise ValueError("If `do_train` is True, then `predict_file` must be specified.")

    if args.do_predict:
        if not args.predict_file:
            raise ValueError("If `do_predict` is True, then `predict_file` must be specified.")

    assert args.task in [
        "dpr", "qa"]

    logger.info("Using {} gpus".format(args.n_gpu))

    if args.bert_name.startswith("bart") or args.bert_name.startswith("t5"):
        args.is_seq2seq = True
    elif args.bert_name.startswith("bert") or args.bert_name.startswith("roberta") or args.bert_name.startswith("albert"):
        args.is_seq2seq = False
    else:
        raise NotImplementedError("Pretrained model not recognized: {}".format(args.bert_name))
    run(args, logger)

if __name__=='__main__':
    main()
