# AmbigQA/AmbigNQ README

## Content
1. [Reference papers](#reference-papers)
2. [AmbigNQ format](#ambignq-format)
3. [NQ-open format](#nq-open-format)
4. [Optional resources](#optional-resources)
5. [Evaluation script](#evaluation-script)
6. [Leaderboard submission guide](#leaderboard-submission-guide)

## Reference papers

If you find AmbigQA/AmbigNQ useful, please cite our paper:
```
```

If you find the original NQ useful, please cite this paper:
```
@article{kwiatkowski2019natural,
  title={Natural questions: a benchmark for question answering research},
  author={Kwiatkowski, Tom and Palomaki, Jennimaria and Redfield, Olivia and Collins, Michael and Parikh, Ankur and Alberti, Chris and Epstein, Danielle and Polosukhin, Illia and Devlin, Jacob and Lee, Kenton and others},
  journal={Transactions of the Association for Computational Linguistics},
  year={2019}
}
```


## AmbigNQ format

The full version contains
- train.json (47M)
- dev.json (17M)
- test.json (359K)

The light version contains
- train_light.json (3.3M)
- dev_light.json (984K)
- test_light.json (188K)

`train.json` and `dev.json` files contains a list of dictionary that represents a single datapoint, with the following keys

- `id` (string): an identifier that is consistent with the original NQ.
- `question` (string): a question. This is identical to the question in the original NQ except we postprocess the string to start with an uppercase and end with a question mark.
- `annotations` (a list of dictionaries): a list of all acceptable outputs, where each output is a dictionary that represents either a single answer or multiple question-answer pairs.
    - `type`: `singleAnswer` or `multipleQAs`
    - (If `type` is `singleAnswer`) `answer`: a list of strings that are all acceptable answer texts
    - (If `type` is `multipleQAs`) `qaPairs`: a list of dictionaries with `question` and `answer`. `question` is a string, and `answer` is a list of strings that are all acceptable answer texts
- `viewed_doc_titles` (a list of strings): a list of titles of Wikipedia pages viewed by crowdworkers during annotations. It is underestimated, as Wikipedia pages viewed through hyperlinks are not included. Note that this should not be the input to a system. It is fine to use it as extra supervision, but please keep in mind that it is underestimated.
- `used_queries` (a list of dictionaries): a list of dictionaries containing the search queries and results that are used by crowdworkers during annotations. Each dictionry contains `query` (a string) and `results` (a list of dictionaries containing `title` and `snipper`). Search results are obtained through Google Search API restricted to Wikipedia (details in the paper). Note that this should not be the input to a system. It is fine to use it as extra supervision.
- `nq_answer` (a list of strings): a list of annotated answers in the original NQ.
- `nq_doc_title` (string): an associated Wikipedia page title in the original NQ.

`test.json` are in the same format except it does not contain `annotations`, `viewed_doc_titles` and `used_queries`.
`{train|dev|test}_light.json` as a lighter version of the full version that only contains `id`, `question` and `annotations` (if it is `train` or `dev`).

## NQ-open format

- nqopen-train.json (9.7M)
- nqopen-dev.json (1.1M)
- nqopen-test.json (489K)
- LICENSE

Each file contains a list of dictionary that represents a single datapoint, with the following keys

- `id` (string): an identifier that is consistent with the original NQ.
- `question` (string): a question.
- `answer` (a list of strings): a list of acceptable answer texts.

## Optional resources

- `enwiki-20200120-pages-articles.xml.bz2`: Wikipedia pages dump from [wikimedia](https://dumps.wikimedia.org/enwiki/20200120/).
- `enwiki-20200120-redirect.sql.gz`: Wikipedia redirect dump from [wikimedia](https://dumps.wikimedia.org/enwiki/20200120/), in case you want to use hyperlink information.
- `latest-all.json.bz2`: Wikidata entities dump (20200120) from [wikimedia](https://dumps.wikimedia.org/wikidatawiki/entities/), in case you want to use Wikidata information.
- `docs.db`: sqlite db that is consistent with [DrQA](https://github.com/facebookresearch/DrQA); containing plain text only, no disambiguation pages
- `docs-html.db`: sqlite db that is consistent with [DrQA](https://github.com/facebookresearch/DrQA), containing html, no disambiguation pages
- (Coming Soon!) Top 100 Wikipedia passages retrieved from Dense Passage Retrieval


## Evaluation script

The following script was tested in Python 3.5 and 3.6.

Step 1. Follow the instruction in [coco-caption](https://github.com/tylin/coco-caption) for setup. If you want to compute F1 answer only, you can skip this.


Step 2. Run the evaluation script via
```
python ambigqa_evaluation_script.py --reference_path {reference data file} --prediction_path {prediction file}
```

The prediction should be a json file with a dictionary that has `id` as a key and a prediction dictionary as a value. A prediction dictionary should be in the following format.

- a list of strings (answers), if you only want to compute F1 answer.
- a list of dictionaries with `question` and `answer` as keys, if you want to compute full metrics.

Example:

To only compute F1 answer
```
{
  "-6631842452804060768": ["1624", "1664"],
  ...
}
```

To compute full metrics
```
{
  "-6631842452804060768": [
    {"question": "When was city of new york city founded with dutch protection?", "answer": "1624"},
    {"question": "When was city of new york city founded with dutch protection?", "answer": "1664"}
  ],
  ...
}
```

## Leaderboard submission guide

Email [Sewon Min](mailto:sewon@cs.washington.edu) with the following:

- test prediction file. As you are not supposed to know which subset of NQ-open test set is AmbigNQ, your file should contain predictions for all NQ-open test examples.
- whether the prediction is in the standard setting or zero-shot setting. i.e. whether the model was trained on AmbigNQ train data or not.
- the name of the model
- [optional] dev prediction file and expected dev results. This is to double-check there is no unexpected problem.
- [optional] the institution, link to the paper/code/demo; can be updated later.


Notes
- Please allow for up to one week ahead of time before getting the test numbers and/or your numbers appear on the leaderboard.
- We limit the number of submissions to be 20 per year and 5 per month.








