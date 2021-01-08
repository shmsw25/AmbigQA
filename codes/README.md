# AmbigQA Baseline Models

**Update as of 07/2020**: Codes for running DPR retrieval, DPR reader and BART reader (SpanSeqGen) on NQ-open and AmbigQA are ready. Stay tuned for Question Generation models!

This repo contains multiple models for open-domain question answering. This code is based on [PyTorch][pytorch] and [HuggingFace Transformers][hf].

This is an original implementation of "Sewon Min, Julian Michael, Hannaneh Hajishirzi, Luke Zettlemoyer. [AmbigQA: Answering Ambiguous Open-domain Questions][ambigqa-paper]. 2020".
```
@article{ min2020ambigqa,
    title={ {A}mbig{QA}: Answering Ambiguous Open-domain Questions },
    author={ Min, Sewon and Michael, Julian and Hajishirzi, Hannaneh and Zettlemoyer, Luke },
    journal={ arXiv preprint arXiv:2004.10645 },
    year={2020}
}
```

This also contains a re-implementation of "Vladimir Karpukhin*, Barlas Oguz*, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, Wen-tau Yih. [Dense Passage Retrieval for Open-domain Question Answering. 2020][dpr-paper]", as part of AmbigQA models. The original implementation can be found [here][dpr-code]. This codebase achieves higher accuracy.
```
@article{ karpukhin2020dense,
    title={ Dense Passage Retrieval for Open-domain Question Answering },
    author={ Karpukhin, Vladimir and Oguz, Barlas and Min, Sewon and Lewis, Patrick and Wu, Ledell and Edunov, Sergey and Chen, Danqi and Yih, Wen-tau },
    journal={ arXiv preprint arXiv:2004.04906 },
    year={2020}
}
```

## Content
1. [Installation](#installation)
2. [Download data](#download-data)
3. Instructions for Training & Testing
    * [DPR Retrieval](#dpr-retrieval)
    * [DPR Reader (Span Selection Model)](#dpr-reader-span-selection-model)
    * [SpanSeqGen (BART Reader)](#spanseqgen-bart-reader)
    * [Finetuning on AmbigQA](#finetuning-on-ambigqa)
4. [Results](#results)
    * [Results with less resources](#results-with-less-resources)
5. [Interactive Demo for Question Answering](#interactive)
6. [Pretrained model checkpoint](#need-preprocessed-data--pretrained-models--predictions)

## Installation
Tested with python 3.6.12
```
pip install torch==1.1.0
pip install git+https://github.com/huggingface/transformers.git@7b75aa9fa55bee577e2c7403301ed31103125a35
pip install wget
```

Also, move `pycocoevalcap` to current directory
```
mv ../pycocoevalcap pycocoevalcap
```

## Download data
Let `data_dir` be a directory to save data.
```
python3 download_data.py --resource data.wikipedia_split.psgs_w100 --output_dir ${data_dir} # provided by original DPR
python3 download_data.py --resource data.wikipedia_split.psgs_w100_20200201 --output_dir ${data_dir} # only for AmbigQA
python3 download_data.py --resource data.nqopen --output_dir ${data_dir}
python3 download_data.py --resource data.gold_passages_info.nq_train --output_dir ${data_dir}
python3 download_data.py --resource data.ambigqa --output_dir ${data_dir}
```

## DPR Retrieval
For training DPR retrieval, please refer to the [original implementation][dpr-code]. This code is for taking checkpoint from the original implementation, and running inference.

Step 1: Download DPR retrieval checkpoint provided by DPR original implementation.
```
python3 download_data.py --resource checkpoint.retriever.multiset.bert-base-encoder --output_dir ${dpr_data_dir}
```

Step 2: Run inference to obtain passage vectors.
```
for i in 0 1 2 3 4 5 6 7 8 9 ; do \ # for parallelization
  python3 cli.py --do_predict --bert_name bert-base-uncased --output_dir out/dpr --dpr_data_dir ${data_dir} --do_predict --task dpr --predict_batch_size 3200 --db_index $i ; \
done
```
- `--predict_batch_size` of 3200 is good for one 32gb GPU.
- `--verbose` to print a progress bar
- This script will tokenize passages in Wikipedia which will takes time. If you want to pre-tokenize first and then launch the job on gpus afterward, please do the following: (1) run the above command with `--do_prepro_only`, and (2) re-run the above command without `--do_prepro_only`.

Each run will take around 1.5 hours with one 32 gpu.

Step 3: Run inference to obtain question vectors and save the retrieval predictions.
```
python3 cli.py --bert_name ber-base-uncased --output_dir out/dpr --dpr_data_dir ${data_dir} --do_predict --task dpr --predict_batch_size 3200 --predict_file data/nqopen/{train|dev|test}.json
```

This script will print out recall rate and save the retrieval results as `out/dpr/{train|dev|test}_predictions.json`.

Tip1: Running this for the first time regardless of the data split will create DPR index and save it, so that the next runs can reuse them. If you do not want to create DPR index multiple times, you can run on one data split first, and run the others afterward. If you have resource to run them in parallel, it may save time to just run all of them in parallel.

Tip2: If you are fine with not printing the recall rate, you can specify `--skip_db_load` to save time. It will then print the recall to be 0, but the prediction file will be saved with no problem.

## DPR Reader (Span Selection Model)

For training on NQ-open, run
```
python3 cli.py --do_train --task qa --output_dir out/nq-span-selection \
    --dpr_data_dir ${data_dir} \
    --train_file data/nqopen/train.json \
    --predict_file data/nqopen/dev.json \
    --bert_name {bert-base-uncased|bert-large-uncased} \
    --train_batch_size 32 --train_M 32 --predict_batch_size 128 \
    --eval_period 2000 --wait_step 10
```

- This script will save preprocessed input data so that it can re-load them once it is created. You might want to preprocess data before launching a job on GPUs.
- `train_batch_size` is # of questions / batch, and `train_M` is # of passages / question. Thus, # of (question, passage) / batch is `train_batch_size * train_M`, which matters for GPU usage. With one 32gb GPU and bert-base-uncased, you can use `train_batch_size * train_M` of 128, as hyperparamters specified in the command above.
- `eval_period` is an interval to test on the dev data. The script will only save the best checkpoint based on the dev data. If you prefer, you can specify `skip_inference` to skip inference on the dev data and save all checkpoints. You can then run the inference script (described next) on the dev data using every checkpoint, and choose the best checkpoint.
- `wait_step` is the number of steps to wait since the best checkpoint, until the training is finished.

When training is done, run the following command for prediction.
```
python3 cli.py --do_predict --task qa --output_dir out/nq-span-selection \
    --dpr_data_dir ${data_dir} \
    --predict_file data/nqopen/{dev|test}.json \
    --bert_name {bert-base-uncased|bert-large-uncased} \
    --predict_batch_size 32
```
This command runs predictions using `out/nq-span-selection/best-model.pt` by default. If you want to run predictions using another checkpoint, please specify its path by `--checkpoint`.


## SpanSeqGen (BART Reader)

Note: this model is different from BART closed-book QA model (implemented [here][bart-closed-book-qa]), because this model reads DPR retrieved passages as input.

First, tokenize passage vectors.
```
for i in 0 1 2 3 4 5 6 7 8 9 ; do \ # for parallelization
  python3 cli.py --bert_name bart-large --output_dir out/dpr --dpr_data_dir ${data_dir} --do_predict --do_prepro_only --task dpr --predict_batch_size 3200 --db_index $i \
done
```

Then, save passage selection from the trained DPR reader:
```
python3 cli.py --do_predict --task qa --output_dir out/nq-span-selection \
    --dpr_data_dir ${data_dir} \
    --predict_file data/nqopen/{train|dev|test}.json \
    --bert_name {bert-base-uncased|bert-large-uncased} \
    --predict_batch_size 32 --save_psg_sel_only
```

Now, train a model on NQ-open by:
```
python3 cli.py --do_train --task qa --output_dir out/nq-span-seq-gen \
    --dpr_data_dir ${data_dir} \
    --train_file data/nqopen/train.json \
    --predict_file data/nqopen/dev.json \
    --psg_sel_dir out/nq-span-selection \
    --bert_name bart-large \
    --discard_not_found_answers \
    --train_batch_size 20 --predict_batch_size 40 \
    --eval_period 2000 --wait_step 10
```

## Finetuning on AmbigQA

In order to experiment on AmbigQA, you can simply repeat the process with NQ-open, with only two differences - (i) specifying `--ambigqa` and `--wiki_2020` at several places and (ii) initialize weights from models trained on NQ-open. Step-by-step instructions are as follows.

First, make DPR retrieval predictions using Wikipedia 2020. You can do so by simply repeating Step 2 and Step 3 of [DPR Retrieval](#dpr-retrieval) with `--wiki_2020` specified.
```
for i in 0 1 2 3 4 5 6 7 8 9 ; do \ # for parallelization
  python3 cli.py --do_predict --bert_name bert-base-uncased --output_dir out/dpr --dpr_data_dir ${data_dir} --do_predict --task dpr --predict_batch_size 3200 --db_index $i --wiki_2020 \
done
python3 cli.py --bert_name ber-base-uncased --output_dir out/dpr --dpr_data_dir ${data_dir} --do_predict --task dpr --predict_batch_size 3200 --predict_file data/nqopen/{train|dev|test}.json --wiki_2020
```

In order to fine-tune DPR span selection model on AmbigQA, run the training command similar to NQ training command, but with `--ambigqa` and `--wiki2020` specified. We also used smaller `eval_period` as the dataset size is smaller.
```
python3 cli.py --do_train --task qa --output_dir out/ambignq-span-selection \
    --dpr_data_dir ${data_dir} \
    --train_file data/ambigqa/train_light.json \
    --predict_file data/ambigqa/dev_light.json \
    --bert_name {bert-base-uncased|bert-large-uncased} \
    --train_batch_size 32 --train_M 32 --predict_batch_size 32 \
    --eval_period 500 --wait_step 10 --topk_answer 3 --ambigqa --wiki_2020
```

In order to fine-tune SpanSeqGen on AmbigQA, first run the inference script over DPR to get highly ranked passages, just like we did on NQ.
```
python3 cli.py --do_predict --task qa --output_dir out/nq-span-selection \
    --dpr_data_dir ${data_dir} \
    --predict_file data/nqopen/{train|dev|test}.json \
    --bert_name {bert-base-uncased|bert-large-uncased} \
    --predict_batch_size 32 --save_psg_sel_only --wiki_2020
```

Next, train SpanSeqGen on AmbigNQ via the following command, which specifies `--ambigqa`, `--wiki_2020` and `--max_answer_length 25`.
```
python3 cli.py --do_train --task qa --output_dir out/ambignq-span-seq-gen \
    --dpr_data_dir ${data_dir} \
    --train_file data/ambigqa/train_light.json \
    --predict_file data/ambigqa/dev_light.json \
    --psg_sel_dir out/nq-span-selection \
    --bert_name bart-large \
    --discard_not_found_answers \
    --train_batch_size 20 --predict_batch_size 40 \
    --eval_period 500 --wait_step 10 --ambigqa --wiki_2020 --max_answer_length 25
```

## Hyperparameter details

**On NQ-open:** For BERT-base, we use `train_batch_size=32, train_M=32` (w/ eight 32GB gpus). For BERT-large, we use `train_batch_size=8, train_M=16` (w/ four 32GB gpus). For BART, we use `train_batch_size=24` (w/ four 32GB gpus). For others, we use default hyperparameters.

**On AmbigQA:** We use `train_batch_size=8` for BERT-base and `train_batch_size=24` for BART. We use `learning_rate=5e-6` for both.

## Results

|   | NQ-open (dev) | NQ-open (test) | AmbigQA zero-shot (dev) | AmbigQA zero-shot (test) | AmbigQA (dev) | AmbigQA (test) |
|---|---|---|---|---|---|---|
|DPR (original implementation)| 39.8 | 41.5 | 35.2/26.5 | 30.1/23.2 | 37.1/28.4 | 32.3/24.8 |
|DPR (this code)| 40.6 | 41.6 | 35.2/23.9 | 29.9/21.4 | 36.8/25.8 | 33.3/23.4 |
|DPR (this code) w/ BERT-large| 43.2 | 44.3 | - | - | - | - |
|SpanSeqGen (reported)| 42.0 | 42.2 | 36.4/24.8 | 30.8/20.7 | 39.7/29.3 | 33.5/24.5 |
|SpanSeqGen (this code)| 43.1 | 45.0 | 37.4/26.1 | 33.2/22.6 | 40.3/29.2 | 35.5/25.8 |

Two numbers on AmbigQA indicate F1 score on all questions and F1 score on questions with multiple QA pairs only.

By default, the models are based on BERT-base and BART-large.

*Note (as of 07/2020)*: Note that numbers are slightly different from those reported in the paper, because numbers in the paper are based on experiments with fairseq. We re-implemented the models with Huggingface Transformers, and were able to obtain similar/better numbers. We will update numbers in the paper of the next version.

*Note*: There happen to be two versions of NQ answers which marginally differ in tokenization methods (e.g. `July 15 , 2020` vs. `July 15, 2020` or `2019 - 2020` vs. `2019--2020`).
Research papers outside Google ([#1][dpr-paper], [#2][ambigqa-paper], [#3][hard-em], [#4][path-retriever], [#5][rag], [#6][colbert], [#7][fusion-decoder], [#8][graph-retriever]) have been using [this version](https://nlp.cs.washington.edu/ambigqa/data/nqopen.zip), and in June 2020 the original NQ/NQ-open authors release the [original version](https://github.com/efficientqa/nq-open) that have been used in research papers from Google ([#1][orqa], [#2][realm], [#3][t5qa]).
We verified that the performance differences are marginal when applying simple postprocessing (e.g. `text.replace(" - ", "-").replace(" : ", ":")`).
The numbers reported here as well as codes follow Google's original version. Compared to the previous version, performance difference is 40.6 (original) vs. 40.3 (previous) vs. 40.7 (union of two) on the dev set and 41.6 (original) vs. 41.7 (previous) vs. 41.8 (union of two) on the test set.
Nonetheless, we advice to use the original version provided by Google in the future.

### Results with less resources

The readers are not very sensitive to hyperparamters (`train_batch_size` and `train_M`). In case you want to experiment with less resources and want to check the reproducibility, here are our results depending on the number of 32gb GPUs.

DPR with BERT-base:
| Num. of 32gb GPU(s) | (`train_batch_size`, `train_M`) | NQ-open (dev) | NQ-open (test) |
|---|---|---|---|
| 1 | (8, 16) | 40.5 | 41.4 |
| 2 | (16, 16) | 40.9 | 41.1 |
| 4 | (16, 32) | 41.2 | 41.1 |
| 8 | (32, 32) | 40.6 | 41.6 |

DPR with BERT-large:
| Num. of 32gb GPU(s) | (`train_batch_size`, `train_M`) | NQ-open (dev) | NQ-open (test) |
|---|---|---|---|
| 2 | (8, 8) | 42.0 | 43.4 |
| 4 | (8, 16) | 43.2 | 44.3 |
| 8 | (16, 16) | 42.2 | 43.2 |


## Interactive

You can run DPR interactively as follows.

```python
from InteractiveDPR import InteractiveDPR
interactive_dpr = InteractiveDPR(dpr_data_dir=path_to_dpr_data_dir reader_checkpoint=path_do_reader_checkpoint)
question = "When did harry potter and the sorcerer's stone movie come out?"
print (interactive_dpr.predict(question, topk_answer=5, only_text=True))
```

For details, please refer to `InteractiveDPR.py`


## Need preprocessed data / pretrained models / predictions?

**DPR**
- [DPR predictions on NQ](https://nlp.cs.washington.edu/ambigqa/models/nq-dpr.zip)

**Question Answering**
Click in order to download checkpoints:
- [DPR Reader trained on NQ (387M)][checkpoint-nq-dpr]
- [DPR Reader (w/ BERT-large) trained on NQ (1.2G)][checkpoint-nq-dpr-large]
- [DPR Reader trained on AmbigNQ (387M)][checkpoint-ambignq-dpr]
- [SpanSeqGen trained on NQ (1.8G)][checkpoint-nq-bart]
- [SpanSeqGen trained on AmbigNQ (1.8G)][checkpoint-ambignq-bart]

**Question Disambiguation**
Coming soon!

[ambigqa-paper]: https://arxiv.org/abs/2004.10645
[dpr-paper]: https://arxiv.org/abs/2004.04906
[dpr-code]: https://github.com/facebookresearch/DPR
[bart-closed-book-qa]: https://github.com/shmsw25/bart-closed-book-qa
[hf]: https://huggingface.co/transformers/
[pytorch]: https://pytorch.org/

[hard-em]: https://arxiv.org/abs/1909.04849
[path-retriever]: https://arxiv.org/abs/1911.10470
[rag]: https://arxiv.org/abs/2005.11401
[fusion-decoder]: https://arxiv.org/abs/2007.01282
[colbert]: https://arxiv.org/abs/2007.00814
[graph-retriever]: https://arxiv.org/abs/1911.03868

[orqa]: https://arxiv.org/abs/1906.00300
[realm]: https://arxiv.org/abs/2002.08909
[t5qa]: https://arxiv.org/abs/2002.08910

[checkpoint-nq-dpr]: https://nlp.cs.washington.edu/ambigqa/models/nq-bert-base-uncased-32-32-0.zip
[checkpoint-nq-dpr-large]: https://nlp.cs.washington.edu/ambigqa/models/nq-bert-large-uncased-16-16-0.zip
[checkpoint-ambignq-dpr]: https://nlp.cs.washington.edu/ambigqa/models/ambignq-bert-base-uncased-8-32-0.zip
[checkpoint-nq-bart]: https://nlp.cs.washington.edu/ambigqa/models/nq-bart-large-24-0.zip
[checkpoint-ambignq-bart]: https://nlp.cs.washington.edu/ambigqa/models/ambignq-bart-large-12-0.zip






