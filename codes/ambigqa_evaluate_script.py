import os
import re
import json
import string
import argparse
import numpy as np
from collections import Counter, defaultdict

from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu

tokenizer = PTBTokenizer()

class QAPairEvaluation(object):

    def __init__(self, reference, prediction, metrics="all"):
        '''
        :param: samples: a list of annotated data
        :param: predictions: a dictionary with id as key and prediction as value
                        prediction can be either
                        - a list of strings
                        - a list of dictionaries with quetion and answer as keys
        '''
        self.reference = reference
        self.prediction = [prediction[sample["id"]] for sample in reference]
        self.metrics = metrics
        METRICS_ANSWER = ["F1 answer"]
        METRICS_QG = ["F1 bleu1", "F1 bleu2", "F1 bleu3", "F1 bleu4", "F1 edit-f1"]

        if metrics=="all" and type(self.prediction[0][0])==str:
            self.metrics = METRICS_ANSWER
        elif metrics=="all":
            self.metrics = METRICS_ANSWER+METRICS_QG

        assert len(set(self.metrics)-set(METRICS_ANSWER)-set(METRICS_QG))==0
        self.QG_METRICS_TO_COMPUTE = [m for m in ["bleu1", "bleu2", "bleu3", "bleu4", "rouge-l", "edit-f1"] if any([metric.endswith(m) for metric in self.metrics])]

        if len(self.QG_METRICS_TO_COMPUTE)>0:
            # if evaluating QG, tokenize prompt question,
            # reference question and predicted question
            data_to_tokenize = {}
            for i, ref in enumerate(self.reference):
                data_to_tokenize["prompt.{}".format(i)] = [{"caption": ref["question"]}]
                for j, annotation in enumerate(ref["annotations"]):
                    if annotation['type']=='multipleQAs':
                        for k, pair in enumerate(annotation['qaPairs']):
                            data_to_tokenize["ref.{}.{}.{}".format(i, j, k)] = \
                                    [{'caption': sent.strip()} for sent in pair["question"].split('|') if len(sent.strip())>0]
            for i, pred in enumerate(self.prediction):
                for j, pair in enumerate(pred):
                    data_to_tokenize["gen.{}.{}".format(i, j)] = [{"caption": pair["question"]}]

            all_tokens =  tokenizer.tokenize(data_to_tokenize)
            for key, values in all_tokens.items():
                values = {'sent': [normalize_answer(value) for value in values]}
                if key.startswith("prompt."):
                    i = key.split(".")[1]
                    self.reference[int(i)]["question"] = values
                elif key.startswith("ref."):
                    i, j, k = key.split('.')[1:]
                    self.reference[int(i)]["annotations"][int(j)]["qaPairs"][int(k)]["question"] = values
                elif key.startswith("gen."):
                    i, j = key.split(".")[1:]
                    self.prediction[int(i)][int(j)]["question"] = values
                else:
                    raise NotImplementedError()

        self.is_multi = [not any([ann["type"]=="singleAnswer" for ann in ref["annotations"]]) \
                      for ref in self.reference]
        self.results = [self.get_all_metrics(idx) for idx in range(len(self.reference))]

    def print_all_metrics(self):
        for metric in self.metrics:
            result = [e[metric] for e in self.results]
            result_multi_only = [e[metric] for e, is_multi in zip(self.results, self.is_multi) \
                                 if is_multi]
            if metric=="F1 answer":
                print ("%s\t%.3f (all)\t%.3f (multi only)" % (metric, np.mean(result), np.mean(result_multi_only)))
            else:
                print ("%s\t%.3f" % (metric, np.mean(result_multi_only)))

    def get_metric(self, metric):
        return np.mean([e[metric] for e in self.results])

    def get_all_metrics(self, idx):
        evaluation = {}
        promptQuestion = self.reference[idx]["question"]
        annotations = self.reference[idx]["annotations"]
        if type(self.prediction[idx][0])==dict:
            # prediction contains a set of question-answer pairs
            predictions = [pair["answer"] for pair in self.prediction[idx]]
            questions = [pair["question"] for pair in self.prediction[idx]]
        else:
            # prediction contains a set of answers
            predictions = self.prediction[idx]
            questions = None

        for annotation in annotations:
            # iterate each annotation and take the maximum metrics
            if annotation['type']=='singleAnswer':
                f1 = get_f1([annotation['answer']], predictions)
                for metric in self.metrics:
                    if metric.startswith('F1'):
                        evaluation[metric] = max(evaluation.get(metric, 0), f1)
            elif annotation['type']=='multipleQAs':
                matching_pairs = []
                evaluation['F1 answer'] = max(evaluation.get("F1 answer", 0),
                                            get_f1([answer['answer'] for answer in annotation['qaPairs']], predictions))
                if questions is None:
                    # skip the below if not evaluating QG
                    continue
                for i, answer in enumerate(annotation["qaPairs"]):
                    for j, prediction in enumerate(predictions):
                        # get every reference-prediction pair with the correct answer prediction
                        em = get_exact_match(answer['answer'], prediction)
                        if em:
                            qg_evals = get_qg_metrics(questions[j],
                                                    answer['question'],
                                                    promptQuestion,
                                                    self.QG_METRICS_TO_COMPUTE)
                            matching_pairs.append((i, j, qg_evals))

                def _get_qg_f1(metric_func):
                    curr_matching_pairs = sorted(matching_pairs, key=lambda x: metric_func(x[2]), reverse=True)
                    occupied_answers = [False for _ in annotation["qaPairs"]]
                    occupied_predictions = [False for _ in predictions]
                    tot = 0
                    # find non-overapping reference-prediction pairs
                    # that match the answer prediction
                    # to get the evaluation score
                    for (i, j, e) in curr_matching_pairs:
                        if occupied_answers[i] or occupied_predictions[j]:
                            continue
                        occupied_answers[i] = True
                        occupied_predictions[j] = True
                        tot += metric_func(e)
                    assert np.sum(occupied_answers)==np.sum(occupied_predictions)
                    return 2 * tot / (len(occupied_answers)+len(occupied_predictions))

                for metric in self.QG_METRICS_TO_COMPUTE:
                    metric_name = "F1 {}".format(metric)
                    if metric_name in self.metrics:
                        e = _get_qg_f1(lambda x: x[metric])
                        evaluation[metric_name] = max(evaluation.get(metric_name, 0), e)
            else:
                raise NotImplementedError()

        assert len(self.metrics)==len(evaluation), (self.metrics, evaluation.keys())
        return evaluation

def get_qg_metrics(generated, question, promptQuestion, metrics):

    evaluation = {}

    # computing bleu scores
    for name, score in zip(['bleu{}'.format(i) for i in range(1, 5)],
                           Bleu(4).compute_score(question, generated)[0]):
        if name in metrics:
            evaluation[name] = score

    # computing edit-f1 score
    if 'edit-f1' in metrics:
        def _get_edits(tokens1, tokens2):
            allCommon = []
            while True:
                commons = list(set(tokens1) & set(tokens2))
                if len(commons)==0:
                    break
                allCommon += commons
                for c in commons:
                    ind1, ind2 = tokens1.index(c), tokens2.index(c)
                    tokens1 = tokens1[:ind1]+tokens1[ind1+1:]
                    tokens2 = tokens2[:ind2]+tokens2[ind2+1:]
            deleted = ["[DELETED]"+token for token in tokens1]
            added = ["[ADDED]"+token for token in tokens2]
            common = ["[FIXED]"+token for token in allCommon]
            return deleted+added #+common

        assert len(generated)==len(promptQuestion)==1
        generated = generated["sent"][0].split(" ")
        promptQuestion = promptQuestion["sent"][0].split(" ")
        prediction = _get_edits(promptQuestion, generated)
        edit_f1 = 0
        for _question in question["sent"]:
            _question = _question.split(" ")
            reference = _get_edits(promptQuestion, _question)
            # now compare the reference edits and predicted edits
            if len(reference)==len(prediction)==0:
                # rarely, reference has no edits after normalization
                # then, if the prediction also has no edits, it gets full score
                edit_f1 = 1
            elif len(reference)==0 or len(prediction)==0:
                # if only one of them has no edits, zero score
                edit_f1 = max(edit_f1, 0)
            else:
                # otherwise, compute F1 score between prediction and reference
                edit_f1 = max(edit_f1, get_f1(prediction, reference, is_equal=lambda x, y: x==y))
        evaluation["edit-f1"] = edit_f1

    assert len(metrics)==len(evaluation)
    return evaluation

def get_exact_match(answers1, answers2):
    if type(answers1)==list:
        if len(answers1)==0:
            return 0
        return np.max([get_exact_match(a, answers2) for a in answers1])
    if type(answers2)==list:
        if len(answers2)==0:
            return 0
        return np.max([get_exact_match(answers1, a) for a in answers2])
    return (normalize_answer(answers1) == normalize_answer(answers2))

def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_f1(answers, predictions, is_equal=get_exact_match, return_p_and_r=False):
    '''
    :answers: a list of list of strings
    :predictions: a list of strings
    '''
    assert len(answers)>0 and len(predictions)>0, (answers, predictions)
    occupied_answers = [False for _ in answers]
    occupied_predictions = [False for _ in predictions]
    for i, answer in enumerate(answers):
        for j, prediction in enumerate(predictions):
            if occupied_answers[i] or occupied_predictions[j]:
                continue
            em = is_equal(answer, prediction)
            if em:
                occupied_answers[i] = True
                occupied_predictions[j] = True
    assert np.sum(occupied_answers)==np.sum(occupied_predictions)
    a, b = np.mean(occupied_answers), np.mean(occupied_predictions)
    if return_p_and_r:
        if a+b==0:
            return 0., 0., 0.
        return 2*a*b/(a+b), float(a), float(b)
    if a+b==0:
        return 0.
    return 2*a*b/(a+b)

def load_reference(reference_path):
    if os.path.exists(reference_path):
        with open(reference_path, "r") as f:
            reference = json.load(f)
        if not (type(reference)==list and \
                all([type(ref)==dict and "id" in ref and "question" in ref and "annotations" in ref and \
                     type(ref["question"])==str and type(ref["annotations"])==list and \
                     all([type(ann)==dict and ann["type"] in ["singleAnswer", "multipleQAs"] for ann in ref["annotations"]]) \
                     for ref in reference])):
            raise Exception("Reference file {} is wrong".format(reference_path))
    else:
        raise Exception("Reference file {} not found".format(reference_path))
    return reference

def load_prediction(prediction_path, ids):
    if os.path.exists(prediction_path):
        with open(prediction_path, "r") as f:
            prediction = json.load(f)
        if str(list(prediction.keys())[0])==int:
            prediction = {str(key):value for key, value in prediction.items()}
        if type(list(prediction.values())[0])==str:
            prediction = {key:[value] for key, value in prediction.items()}
        if not (type(prediction)==dict and \
                len(ids-set(prediction.keys()))==0):
            raise Exception("Prediction file {} is wrong".format(prediction_path))
        if not (all([type(pred)==list for pred in prediction.values()]) and \
                (all([type(p)==str for pred in prediction.values() for p in pred]) or \
                 all([type(p)==dict and "question" in p and "answer" in p \
                      and type(p["question"])==type(p["answer"])==str for pred in prediction.values() for p in pred]))):
            raise Exception("Prediction file {} has a wrong format".format(prediction_path))
    else:
        raise Exception("Prediction file {} not found".format(prediction_path))
    return prediction

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference_path', type=str, required=True)
    parser.add_argument('--prediction_path', type=str, required=True)
    args = parser.parse_args()

    reference = load_reference(args.reference_path)
    ids = set([d["id"] for d in reference])
    prediction = load_prediction(args.prediction_path, ids)
    evaluation = QAPairEvaluation(reference, prediction)
    evaluation.print_all_metrics()


