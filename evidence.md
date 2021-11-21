# AmbigQA/AmbigNQ README

We released semi-oracle evidence passages for researchers interested in multi-answer extraction and disambiguation rather than retrieval. This document describes how they are obtained, statistics and the upperbound when using these evidence passages.


## Content
1. [General information about the data](#general-information-about-the-data)
    * [Data format](#data-format)
    * [When to use this data](#when-to-use-this-data)
    * [Statistics and performance upperbound](#statistics-and-performance-upperbound)
2. [Data Creation](#data-creation)


## General information about the data


The number of Wikipedia articles per question is 3.0 on average.

### Data Format

The json file is a list, which i-th item is a dictionary containing `id`, `question`, `annotations` (as in the original AmbigQA data) as well as `articles_plain_text` and `articles_html_text`. `articles_plain_text` is a list of articles in plain text, such as:
```python
[
  "# Dexter (season 1)\n\nThe first season of Dexter is an adaptation of Jeff Lindsay's first novel in a series of the same name, Darkly Dreaming Dexter. ...",
  "# Chrisstian Camargo\n\nChristian Camargo is an American actor, producer, writer and director. ... ## Early years\n\nCamargo was born ...",
  "# List of Dexter characters\n\nThis is a list of characters ... * Michael C. Hall\n* Maxwell Huckabee (age 3) * Nicholas Vigneau (young Dexter, season 7) ..."
]
```
`article_html_text` is a list of articles in an html format, such as:
```python
[
  "<h1>Dexter (season 1)\n\nThe first season of Dexter is an adaptation of Jeff Lindsay's first novel in a series of the same name, Darkly Dreaming Dexter. ...",
  "<h1>Chrisstian Camargo</h1>\n\nChristian Camargo is an American actor, producer, writer and director. ... <h2>Early years</h2>\n\nCamargo was born ...",
  "<h1>List of Dexter characters</h1>\n\nThis is a list of characters ... <ul><li>Michael C. Hall</li><li>Maxwell Huckabee (age 3)</li><li>Nicholas Vigneau (young Dexter, season 7)</li> ..."
]
```

### When to use this data

We recommend using this data if you want to focus on multi-answer extraction and disambiguation given evidence text.
The end-to-end QA model is supposed to retrieve evidence text, but evidence retrieval itself is a very difficult problem and current retrieval models are not good at retrieving high-coverage evidence text (reference: [this paper](https://arxiv.org/abs/2104.08445)). While we encourage making progress in the retrieval part, we are releasing this semi-oracle evidence data so that the progress in the subsequent part is not blocked by the progress in retrieval.


While the size of the evidence text can be a variable in the end-to-end QA model, we set the size of the semi-oracle evidence to be approximately 10,000 words, following much of recent work in QA that uses 100 passages * 100 words per passage.


### Statistics and performance upperbound

#### Number of Wikipedia articles per question
|   | Mean | Median | 90 Percentile | 95 Percentile |
|---|---|---|---|---|
| Train | 3.0 | 3.0 | 3.0 | 3.0 |
| Dev   | 3.0 | 3.0 | 3.0 | 3.0 |
| Test  | 3.0 | 3.0 | 3.0 | 3.0 |

#### Number of tokens per question
(based on the plain text, white space tokenization)
|   | Mean | Median | 90 Percentile | 95 Percentile |
|---|---|---|---|---|
| Train | 9344.4 | 7532.0 | 18420.5 | 22564.3 |
| Dev   | 9371.3 | 7561.0 | 18560.7 | 22690.8 |
| Test  | 9530.3 | 7759.5 | 19001.2 | 23357.6 |

#### Answer coverage and performance upperbound
(Performance upperbound is the same for both answer F1 and QG F1)

|   | Macro-Avg coverage | Perf upperbound (all) | Perf upperbound (multi-only) |
|---|---|---|---|
| Train | 78.2 | 80.1 | 77.1 |
| Dev   | 84.4 | 86.6 | 82.2 |
| Test  | 83.0 | 85.6 | 81.3 |


#### Distributions of the number of covered answers (%)

|   | 0 | 1 | 2 | 3 | 4+ |
|---|---|---|---|---|---|
| Train | 10.1 | 62.8 | 33.6 | 23.8 | 10.1 |
| Dev   | 15.7 | 58.5 | 42.4 | 30.4 | 15.7 |
| Test  | 18.8 | 56.1 | 45.6 | 36.0 | 18.8 |


## Data Creation

We use the Wikipedia dump of 02/01/2020, which is the same one as used in the [AmbigQA paper](https://arxiv.org/abs/2004.10645). We preprocess the dump so that each article includes headers, plain text and lists (tables and infoboxes are excluded). We excluded disambiguation pages, following prior work (DrQA, DPR and more).

We look up the annotator interactive logs, and find positive articles and negative articles as follows.
* Positive articles: we examine articles that anotator clicked (if they clicked a disambiguation page, articles that are linked to the disambiguation page), and include articles that contain any valid answer as positive articles.
* Negative articles: we include all articles that annotators have seen (including just titles). This includes articles that are result of the search engine and all articles linked to the disambiguation page. Among those, articles that do not contain the valid answers are considered as negative articles.

Once we obtain positive articles and negative articles, we create a set of articles by (1) first including all positive articles, and (2) if the number of positive articles is less than 3, sampling negative articles as follows.
1. Create a BM25 index using all positive and negative articles.
2. Compute BM25 scores of each article using the question as a query.
3. Compute a weight probability using a softmax of BM25 scores.
4. Sample articles based on the weight probability, until the number of unique articles is 3.












