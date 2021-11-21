# AmbigQA/AmbigNQ README

This is the repository documenting the paper
[AmbigQA: Answering Ambiguous Open-domain Questions](https://arxiv.org/abs/2004.10645) (EMNLP 2020)
by Sewon Min, Julian Michael, Hannaneh Hajishirzi, and Luke Zettlemoyer.

* [Website](https://nlp.cs.washington.edu/ambigqa)
* Read the [paper](https://arxiv.org/abs/2004.10645)
* Download the dataset: [AmbigNQ light ver.](https://nlp.cs.washington.edu/ambigqa/data/ambignq_light.zip) / [AmbigNQ full ver.](https://nlp.cs.washington.edu/ambigqa/data/ambignq.zip) / [NQ-open](https://nlp.cs.washington.edu/ambigqa/data/nqopen.zip)
* **Update (07/2020)**: Try running [baseline codes][codes]
* **Update (11/2021)**: We released semi-oracle evidence passages for researchers interested in multi-answer extraction and disambiguation rather than retrieval. Please read [evidence.md](https://github.com/shmsw25/AmbigQA/tree/master/evidence.md) for details.

## Content
1. [Citation](#citation)
2. [Dataset Contents](#dataset-contents)
    * [AmbigNQ](#ambignq)
    * [AmbigNQ with evidence articles](#ambignq-with-evidence-articles)
    * [NQ-open](#nq-open)
    * [Additional resources](#additional-resources)
3. [Evaluation script](#evaluation-script)
4. [Baseline codes](#baseline-codes)
5. [Leaderboard submission guide](#leaderboard-submission-guide)

## Citation

If you find the AmbigQA task or AmbigNQ dataset useful, please cite our paper:
```
@inproceedings{ min2020ambigqa,
    title={ {A}mbig{QA}: Answering Ambiguous Open-domain Questions },
    author={ Min, Sewon and Michael, Julian and Hajishirzi, Hannaneh and Zettlemoyer, Luke },
    booktitle={ EMNLP },
    year={2020}
}
```

Please also make sure to credit and cite the creators of Natural Questions,
the dataset which we built ours off of:
```
@article{ kwiatkowski2019natural,
  title={ Natural questions: a benchmark for question answering research},
  author={ Kwiatkowski, Tom and Palomaki, Jennimaria and Redfield, Olivia and Collins, Michael and Parikh, Ankur and Alberti, Chris and Epstein, Danielle and Polosukhin, Illia and Devlin, Jacob and Lee, Kenton and others },
  journal={ Transactions of the Association for Computational Linguistics },
  year={ 2019 }
}
```


## Dataset Contents

### AmbigNQ

[Click here to download the light version of the data (1.1M)](https://nlp.cs.washington.edu/ambigqa/data/ambignq_light.zip).
[Click here to download the full version of the data (18M)](https://nlp.cs.washington.edu/ambigqa/data/ambignq.zip).

We provide two distributions of our new dataset AmbigNQ: a `full` version with all annotation metadata
and a `light` version with only inputs and outputs.

The full version contains
- train.json (47M)
- dev.json (17M)

The light version contains
- train_light.json (3.3M)
- dev_light.json (977K)

`train.json` and `dev.json` files contain a list of dictionary that represents a single datapoint, with the following keys

- `id` (string): an identifier for the question, consistent with the original NQ dataset.
- `question` (string): a question. This is identical to the question in the original NQ except we postprocess the string to start uppercase and end with a question mark.
- `annotations` (a list of dictionaries): a list of all acceptable outputs, where each output is a dictionary that represents either a single answer or multiple question-answer pairs.
    - `type`: `singleAnswer` or `multipleQAs`
    - (If `type` is `singleAnswer`) `answer`: a list of strings that are all acceptable answer texts
    - (If `type` is `multipleQAs`) `qaPairs`: a list of dictionaries with `question` and `answer`. `question` is a string, and `answer` is a list of strings that are all acceptable answer texts
- `viewed_doc_titles` (a list of strings): a list of titles of Wikipedia pages viewed by crowdworkers during annotations. This is an underestimate, since Wikipedia pages viewed through hyperlinks are not included. Note that this should not be the input to a system. It is fine to use it as extra supervision, but please keep in mind that it is an underestimate.
- `used_queries` (a list of dictionaries): a list of dictionaries containing the search queries and results that were used by crowdworkers during annotations. Each dictionary contains `query` (a string) and `results` (a list of dictionaries containing `title` and `snippet`). Search results are obtained through the Google Search API restricted to Wikipedia (details in the paper). Note that this should not be the input to a system. It is fine to use it as extra supervision.
- `nq_answer` (a list of strings): the list of annotated answers in the original NQ.
- `nq_doc_title` (string): an associated Wikipedia page title in the original NQ.

`{train|dev}_light.json` are formatted the same way, but only contain `id`, `question` and `annotations`.


### AmbigNQ with evidence articles

[Click here to download the data (3.9M)](https://nlp.cs.washington.edu/ambigqa/data/ambignq_with_evidence_articles.zip).

Please read [evidence.md](https://github.com/shmsw25/AmbigQA/tree/master/evidence.md) for details.

The evidence version contains
- train_with_evidence_articles.json (1.2G)
- dev_with_evidence_articles.json (241M)
- test_with_evidence_articles_without_answers.json (245M)

They contain a list of dictionary that represents a single datapoint, just as the above. In addition to `id`, `question` and `annotations` (omitted in the test data), each dictionary contains

- `articles_plain_text`: a list of articles in the plain text.
- `articles_html_text`: a list of articles in the HTML text.


### NQ-open

[Click here to download the data (3.9M)](https://nlp.cs.washington.edu/ambigqa/data/nqopen.zip).


We release our split of NQ-open, for comparison and use as weak supervision:

- nqopen-train.json (9.7M)
- nqopen-dev.json (1.1M)
- nqopen-test.json (489K)

Each file contains a list of dictionaries representing a single datapoint, with the following keys

- `id` (string): an identifier that is consistent with the original NQ.
- `question` (string): a question.
- `answer` (a list of strings): a list of acceptable answer texts.

### Additional resources

- `docs.db`: sqlite db that is consistent with [DrQA](https://github.com/facebookresearch/DrQA); containing plain text only, no disambiguation pages
- `docs-html.db`: sqlite db that is consistent with [DrQA](https://github.com/facebookresearch/DrQA), containing html, no disambiguation pages
- Top 100 Wikipedia passages retrieved from Dense Passage Retrieval

## Evaluation script

The evaluation script is [here](https://github.com/shmsw25/AmbigQA/blob/master/ambigqa_evaluate_script.py).
It has been tested on Python 3.5 and 3.6.

Step 1. Follow the instruction in [coco-caption](https://github.com/tylin/coco-caption) for setup. If you want to compute F1 answer only, you can skip this.

Step 2. Run the evaluation script via
```
python ambigqa_evaluation_script.py --reference_path {reference data file} --prediction_path {prediction file}
```

The prediction should be a json file with a dictionary that has `id` as a key and a prediction object as a value. A prediction object should be in the following format.

- a list of strings (answers), if you only want to compute answer F1.
- a list of dictionaries with "question" and "answer" as keys, if you want to compute full metrics.

Example:

To only compute answer F1:
```
{
  "-6631842452804060768": ["1624", "1664"],
  ...
}
```

To compute full metrics:
```
{
  "-6631842452804060768": [
    {"question": "When was city of new york city founded with dutch protection?", "answer": "1624"},
    {"question": "When was city of new york city founded and renamed with english name?", "answer": "1664"}
  ],
  ...
}
```

## Baseline codes

Try running [baseline codes][codes] (instructions in its README), which includes DPR retrieval, DPR reader and SpanSeqGen. This includes codes and scripts for both NQ-open and AmbigNQ.


## Leaderboard submission guide

Create a prediction file using the questions on NQ-open test data, and email it to [Sewon Min](mailto:sewon@cs.washington.edu).

Please make sure you include the following in the email:

- test prediction file. Make sure that the format is in line with the official evaluation script. As you are not supposed to know which subset of NQ-open test set is AmbigNQ, your file should contain predictions for all NQ-open test examples.
- whether the prediction is in the standard setting or zero-shot setting, i.e. whether the model was trained on AmbigNQ train data or not.
- the name of the model
- [optional] dev prediction file and expected dev results. This is to double-check there is no unexpected problem.
- [optional] the institution, and link to the paper/code/demo. They can be updated later.


Notes
- Models will be sorted by `F1 answer (all) + F1 edit-f1` (standard) or `F1 answer (all)` (zero-shot).
- Please allow for up to one week ahead of time before getting the test numbers and/or your numbers appear on the leaderboard.
- We limit the number of submissions to be 20 per year and 5 per month.


[codes]: https://github.com/shmsw25/AmbigQA/tree/master/codes






