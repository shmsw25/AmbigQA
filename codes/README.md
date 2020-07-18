# AmbigQA-Models

*Update as of 07/2020*: Codes for running DPR retrieval, DPR reader and BART reader (SpanSeqGen) on NQ-open and AmbigQA are ready. Stay tuned for Question Generation models!

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

This also contains a re-implementation of "Vladimir Karpukhin*, Barlas Oguz*, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, Wen-tau Yih. [Dense Passage Retrieval for Open-domain Question Answering. 2020][dpr-paper]", as part of AmbigQA models. The original implementation can be found [here][dpr-code]. This codebase achieves higher accuracy and is more memory efficient (best result achieved with two 32gb GPUs instead of eight; see aggregated results in the last section of this README).
```
@article{ karpukhin2020dense,
    title={ Dense Passage Retrieval for Open-domain Question Answering },
    author={ Karpukhin, Vladimir and Oguz, Barlas and Min, Sewon and Lewis, Patrick and Wu, Ledell and Edunov, Sergey and Chen, Danqi and Yih, Wen-tau },
    journal={ arXiv preprint arXiv:2004.04906 },
    year={2020}
}
```

Please see [Results section](#Aggregated-results) in this README to compare various models & see updated numbers.

## Installation

```
pip install torch==1.1.0
pip install git+https://github.com/huggingface/transformers.git@7b75aa9fa55bee577e2c7403301ed31103125a35
```

## Download data
Let `data_dir` be a directory to save data.
```
python3 download_data.py --resource data.wikipedia_split --output_dir ${data_dir}
python3 download_data.py --resource data.nqopen --output_dir ${data_dir}
python3 download_data.py --resource data.ambigqa --output_dir ${data_dir}
```

## DPR Retrieval

For training DPR retrieval, please refer to the [original implementation][dpr-code]. This code is for taking checkpoint from the original implementation, and running inference.

Step 1: Download Wikipedia DB and DPR retrieval checkpoint provided by DPR original implementation.
```
python3 download_data.py --resource data.wikipedia_split --output_dir ${data_dir} # Wikipedia DB
python3 download_data.py --resource checkpoint.retriever.multi.bert-base-encoder --output_dir ${dpr_data_dir} # retrieval checkpoint
```

Step 2: Run inference to obtain passage vectors.
```
for i in 0 1 2 3 4 5 6 7 8 9 ; do \ # for parallelization
  python3 cli.py --bert_name bert-base-uncased --output_dir out/dpr --do_predict --task dpr --predict_batch_size 3200 --db_index $i \
done
```
- `--predict_batch_size` of 3200 is good for one 32gb GPU.
- `--verbose` to print a progress bar
- This script will tokenize passages in Wikipedia which will takes time. If you want to pre-tokenize first and then launch the job on gpus afterward, please do the following: (1) run the above command first, (2) when the log prints "Finish loading ### bert tokenized data", stop the job, and (3) re-run the above command with specifying `--skip_db_load`.

Each run will take around 1.5 hours with one 32 gpu.

Step 3: Run inference to obtain question vectors and save the retrieval predictions.
```
python3 cli.py --bert_name ber-base-uncased --output_dir out/dpr --do_predict --task dpr --predict_batch_size 3200 --predict_file data/nqopen-{train|dev|test}.json
```

This script will print out recall rate and save the retrieval results as `out/dpr/{train|dev|test|}_predictions.json`.

Tip1: Running this for the first time regardless of the data split will create DPR index and save it, so that the next runs can reuse them. If you do not want to create DPR index multiple times, you can run on one data split first, and run the others afterward. If you have resource to run them in parallel, it may save time to just run all of them in parallel.

Tip2: If you are fine with not printing the recall rate, you can specify `--skip_db_load` to save time. It will then print the recall to be 0, but the prediction file will be saved with no problem.

## DPR Reader (Span Selection Model)

For training on NQ-open, run
```
python3 cli.py --do_train --task qa --output_dir out/nq-span-selection \
    --train_file data/nqopen-train.json \
    --predict_file data/nqopen-dev.json \
    --bert_name {bert-base-uncased|bert-large-uncased} \
    --train_batch_size 16 --train_M 8 --predict_batch_size 32 \
    --eval_period 2000 --wait_step 10
```

- This script will save preprocessed input data so that it can re-load them once it is created. You might want to preprocess data before launching a job on GPUs.
- `train_batch_size` is # of questions / batch, and `train_M` is # of passages / question. Thus, # of (question, passage) / batch is `train_batch_size * train_M`, which matters for GPU usage. With one 32gb GPU and bert-base-uncased, you can use `train_batch_size * train_M` of 128, as hyperparamters specified in the command above.
- `eval_period` is an interval to test on the dev data. The script will only save the best checkpoint based on the dev data. If you prefer, you can specify `skip_inference` to skip inference on the dev data and save all checkpoints. You can then run the inference script (described next) on the dev data using every checkpoint, and choose the best checkpoint.
- `wait_step` is the number of steps to wait since the best checkpoint, until the training is finished.

When training is done, run the following command for prediction.
```
python3 cli.py --do_train --task qa --output_dir out/nq-span-selection \
    --predict_file data/nqopen-{dev|test}.json \
    --bert_name {bert-base-uncased|bert-large-uncased} \
    --predict_batch_size 32 --eval_period 500
```
This command runs predictions using `out/nq-span-selection/best-model.pt` by default. If you want to run predictions using another checkpoint, please specify its path by `--checkpoint`.


## BART Reader (SpanSeqGen Model)

Note: this model is different from BART closed-book QA model (implemented [here][bart-closed-book-qa]), because this model reads DPR retrieved passages as input.

First, tokenize passage vectors.
```
for i in 0 1 2 3 4 5 6 7 8 9 ; do \ # for parallelization
  python3 cli.py --bert_name bart-large --output_dir out/dpr --do_predict --task dpr --predict_batch_size 3200 --db_index $i \
done
```

Then, save passage selection from the trained DPR reader:
```
python3 cli.py --do_train --task qa --output_dir out/nq-span-selection \
    --predict_file data/nqopen-{train|dev|test}.json \
    --bert_name {bert-base-uncased|bert-large-uncased} \
    --predict_batch_size 32 --save_psg_sel_only
```

Now, train a model on NQ-open by:
```
python3 cli.py --do_train --task qa --output_dir out/nq-span-seq-gen \
    --train_file data/nqopen-train.json \
    --predict_file data/nqopen-dev.json \
    --psgs_sel_dir out/nq-span-selection \
    --bert_name bart-large \
    --discard_not_found_answers \
    --train_batch_size 20 --predict_batch_size 40 \
    --eval_period 2000 --wait_step 10
```

Next, finetune this model to train on AmbigNQ.

```
python3 cli.py --do_train --task qa --output_dir out/ambignq-span-seq-gen \
    --train_file data/train.json \
    --predict_file data/dev.json \
    --psgs_sel_dir out/nq-span-selection \
    --bert_name bart-large \
    --discard_not_found_answers \
    --train_batch_size 20 --predict_batch_size 40 \
    --eval_period 500 --wait_step 10
```

## Hyperparameter details

**On NQ-open:** For BERT-base, we use `train_batch_size=32, train_M=32` (w/ eight 32GB gpus). For BERT-large, we use `train_batch_size=8, train_M=16` (w/ four 32GB gpus). For BART, we use `train_batch_size=24` (w/ four 32GB gpus). For others, we use default hyperparameters.

**On AmbigQA:** For BART, we use `train_batch_size=24, learning_rate=1e-6` (w/ four 32FB gpus).

## Aggregated Results

|   | NQ-open (dev) | NQ-open (test) | AmbigQA zero-shot (dev) | AmbigQA zero-shot (test) | AmbigQA (dev) | AmbigQA (test) |
|---|---|---|---|---|---|---|
|DPR (original implementation)| 39.8 | 41.5 | 35.2/26.5 | 30.1/23.2 | 37.1/28.4 | 32.3/24.8 |
|DPR (this code)| 40.6 | 41.6 | TODO | TODO | TODO | TODO |
|DPR (this code) w/ BERT-large| 43.2 | 44.3 | - | - | - | - |
|SpanSeqGen (reported)| 42.0 | 42.2 | 36.4/24.8 | 30.8/20.7 | 39.7/29.3 | 33.5/24.5 |
|SpanSeqGen (this code)| 43.1 | 45.0 | 35.3/25.9 | 35.8/23.4 | 40.5/27.8 | 36.2/24.6 |

(By default, the models are based on BERT-base and BART-large.)

*Note (as of 07/2020)*: Note that numbers are slightly different from those reported in the paper, because numbers in the paper are based on experiments with fairseq. We re-implemented the models with Huggingface Transformers, and were able to obtain similar/better numbers. We will update numbers in the paper of the next version.

*Note*: There happen to be two versions of NQ answers which marginally differ in tokenization methods (e.g. `July 15 , 2020` vs. `July 15, 2020` or `2019 - 2020` vs. `2019--2020`).
Research papers outside Google ([#1][dpr-paper], [#2][ambigqa-paper], [#3][hard-em], [#4][path-retriever], [#5][rag], [#6][colbert], [#7][fusion-decoder], [#8][graph-retriever]) have been using [this version](https://nlp.cs.washington.edu/ambigqa/data/nqopen.zip), and in June 2020 the original NQ/NQ-open authors release the [original version](https://github.com/efficientqa/nq-open) that have been used in research papers from Google ([#1][orqa], [#2][realm], [#3][t5qa]).
We verified that the performance differences are marginal when applying simple postprocessing (e.g. `text.replace(" - ", "-")`).
The numbers reported here as well as codes follow Google's original version. Compared to the previous version, performance difference is 40.6 (original) vs. 40.3 (previous) vs. 40.7 (union of two) on the dev set and 41.6 (original) vs. 41.7 (previous) vs. 41.8 (union of two) on the test set.
Nonetheless, we advice to use the original version provided by Google in the future.

## Interactive

You can run DPR interactively as follows.

```python
from InteractiveDPR import InteractiveDPR
interactive_dpr = InteractiveDPR(dpr_data_dir=path_to_dpr_data_dir reader_checkpoint=path_do_reader_checkpoint)
question = "When did harry potter and the sorcerer's stone movie come out?"
print (interactive_dpr.predict(question, topk_answer=5, only_text=True))
```

For details, please refer to `InteractiveDPR.py`

Currently it is only supporting DPR retrieval + DPR reader (Span selection reader). Please stay tuned for SpanSeqGen.

## Need preprocessed data / pretrained models / predictions?

Due to the storage limit, we cannot provide all preprocessed data / pretrained models. We will provide them based on requests, so please leave issues.


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



