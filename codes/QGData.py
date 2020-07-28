import os
import json
import gzip
import re
import pickle as pkl
import string
import numpy as np
from tqdm import tqdm
from collections import Counter, defaultdict

import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler

from QAData import QAData, AmbigQAData
from DataLoader import MySimpleQADataset, MySimpleQADatasetForPair, MyDataLoader
from util import decode_span_batch

# for evaluation
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from ambigqa_evaluate_script import normalize_answer, get_exact_match, get_f1, get_qg_metrics
from pycocoevalcap.bleu.bleu import Bleu

class QGData(QAData):
    def __init__(self, logger, args, data_path, is_training, passages=None):
        super(QGData, self).__init__(logger, args, data_path, is_training, passages)
        self.qg_tokenizer = PTBTokenizer()
        self.metric = "Bleu"
        if not self.is_training:
            self.qg_tokenizer = PTBTokenizer()

    def load_dpr_data(self):
        dpr_retrieval_path = "out/dpr/{}_predictions.json".format(
            self.data_type+"_20200201" if self.args.wiki_2020 else self.data_type)
        postfix = self.tokenizer.__class__.__name__.replace("zer", "zed")
        dpr_tokenized_path = dpr_retrieval_path.replace(".json", "_{}_qg.json".format(postfix))
        assert "Bart" in postfix
        return self.load_dpr_data_bart(dpr_retrieval_path, dpr_tokenized_path)

    def load_dpr_data_bart(self, dpr_retrieval_path, dpr_tokenized_path):
        self.logger.info("{}\n{}".format(dpr_retrieval_path, dpr_tokenized_path))
        if os.path.exists(dpr_tokenized_path):
            self.logger.info("Loading DPR data from {}".format(dpr_tokenized_path))
            with open(dpr_tokenized_path, "r") as f:
                self.tokenized_data = json.load(f)
        else:
            self.logger.info("Start processing DPR data")
            if self.passages.tokenized_data is None:
                self.passages.load_tokenized_data("bart", all=True)
            if "train_for_inference" not in dpr_retrieval_path:
                dpr_retrieval_path = dpr_retrieval_path.replace("train", "train_for_inference")
            with open(dpr_retrieval_path, "r") as f:
                dpr_passages = json.load(f)
                assert len(dpr_passages)==len(self)
            assert self.args.psg_sel_dir is not None
            psg_sel_fn = os.path.join(self.args.psg_sel_dir,
                                      "{}{}_psg_sel.json".format(
                                          self.data_type.replace("train", "train_for_inference") if "for_inference" not in self.data_type else self.data_type,
                                          "_20200201" if self.args.wiki_2020 else ""))
            self.logger.info("Loading passage selection from DPR reader: {}".format(psg_sel_fn))
            with open(psg_sel_fn, "r") as f:
                fg_passages = json.load(f)
                assert len(fg_passages)==len(dpr_passages)
                dpr_passages = [[psgs[i] for i in fg_psgs] for psgs, fg_psgs in zip(dpr_passages, fg_passages)]
            input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, metadata = self.tokenized_data
            assert len(dpr_passages)==len(input_ids)==len(attention_mask)
            bos_token_id = self.tokenizer.bos_token_id

            def _get_tokenized_answer(idx):
                tokens = self.tokenized_data[2][idx]
                if 0 in self.tokenized_data[3][idx]:
                    tokens = tokens[:self.tokenized_data[3][idx].index(0)]
                assert tokens[0]==tokens[1]==self.tokenizer.bos_token_id and tokens[-1]==self.tokenizer.eos_token_id
                return tokens[2:-1]

            def _included(tokens, curr_input_ids, end_of_answer):
                for i in range(end_of_answer, 1024-len(tokens)+1):
                    if curr_input_ids[i:i+len(tokens)]==tokens:
                        return True
                return False

            has_valid = []
            new_input_ids, new_attention_mask, new_decoder_input_ids, new_decoder_attention_mask, new_metadata = [], [], [], [], []
            for idx, (curr_input_ids, curr_attention_mask, curr_metadata, dpr_ids) in enumerate(zip(
                    input_ids, attention_mask, metadata, dpr_passages)):
                dpr_input_ids = [self.passages.tokenized_data["input_ids"][_id] for _id in dpr_ids]
                dpr_attention_mask = [self.passages.tokenized_data["attention_mask"][_id] for _id in dpr_ids]

                # create multiple inputs
                answer_input_ids_list, answer_attention_mask_list, is_valid_list = [], [], []
                for answer_idx in range(*curr_metadata):
                    end_of_answer = decoder_input_ids[answer_idx].index(self.tokenizer.eos_token_id)+1
                    answer_input_ids = decoder_input_ids[answer_idx][:end_of_answer]
                    answer_attention_mask = decoder_attention_mask[answer_idx][:end_of_answer]
                    offset = 0
                    while len(answer_input_ids)<1024:
                        assert dpr_input_ids[offset][0] == bos_token_id
                        assert len(dpr_input_ids[offset])==len(dpr_attention_mask[offset])
                        assert np.sum(dpr_attention_mask[offset])==len(dpr_attention_mask[offset])
                        answer_input_ids += dpr_input_ids[offset][1:]
                        answer_attention_mask += dpr_attention_mask[offset][1:]
                        offset += 1
                    assert len(answer_input_ids)==len(answer_attention_mask)
                    answer_input_ids_list.append(answer_input_ids[:1024])
                    answer_attention_mask_list.append(answer_attention_mask[:1024])
                    is_valid_list.append(_included(
                        decoder_input_ids[answer_idx][2:end_of_answer-1],
                        answer_input_ids, end_of_answer))

                has_valid.append(any(is_valid_list))
                if self.is_training:
                    if not any(is_valid_list):
                        is_valid_list = [True for _ in is_valid_list]
                    new_metadata.append((len(new_input_ids), len(new_input_ids)+sum(is_valid_list)))
                    new_input_ids += [answer_input_ids for answer_input_ids, is_valid in
                                      zip(answer_input_ids_list, is_valid_list) if is_valid]
                    new_attention_mask += [answer_attention_mask for answer_attention_mask, is_valid in
                                           zip(answer_attention_mask_list, is_valid_list) if is_valid]
                else:
                    index = is_valid_list.index(True) if any(is_valid_list) else 0
                    new_metadata.append((len(new_input_ids), len(new_input_ids)+1))
                    new_input_ids.append(answer_input_ids_list[index])
                    new_attention_mask.append(answer_attention_mask_list[index])
                new_decoder_input_ids.append(curr_input_ids)
                new_decoder_attention_mask.append(curr_attention_mask)

            assert len(new_input_ids)==len(new_attention_mask)==new_metadata[-1][-1]
            self.tokenized_data = [new_input_ids, new_attention_mask, new_decoder_input_ids, new_decoder_attention_mask, new_metadata]
            with open(dpr_tokenized_path, "w") as f:
                json.dump(self.tokenized_data, f)
            self.logger.info("Finish saving tokenized DPR data at {}".format(dpr_tokenized_path))
            self.logger.info("%.1f%% questions have at least one answer mentioned in passages" % (100*np.mean(has_valid)))


    def load_dataset(self, tokenizer, do_return=False):
        if self.tokenized_data is None:
            self.load_tokenized_data(tokenizer)
        input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, metadata = self.tokenized_data
        self.dataset = MySimpleQADataset(input_ids,
                                            attention_mask,
                                            decoder_input_ids if self.is_training or self.args.nq_answer_as_prefix else None,
                                            decoder_attention_mask if self.is_training or self.args.nq_answer_as_prefix else None,
                                            in_metadata=metadata,
                                            out_metadata=None,
                                            is_training=self.is_training,
                                            answer_as_prefix=self.args.nq_answer_as_prefix)
        self.logger.info("Loaded {} examples from {} data".format(len(self.dataset), self.data_type))

        if do_return:
            return self.dataset

    def load_dataloader(self, do_return=False):
        self.dataloader = MyDataLoader(self.args, self.dataset, is_training=self.is_training)
        if do_return:
            return self.dataloader

    def evaluate(self, predictions, n_paragraphs=None):
        assert len(predictions)==len(self), (len(predictions), len(self))
        bleu = []

        # first, tokenize
        data_to_tokenize = {}
        for i, (d, pred) in enumerate(zip(self.data, predictions)):
            data_to_tokenize["ref.{}".format(i)] = [{"caption": d["question"]}]
            data_to_tokenize["gen.{}".format(i)] = [{"caption": pred if type(pred)==str else pred[0]}]
        all_tokens = self.qg_tokenizer.tokenize(data_to_tokenize)
        for i in range(len(self.data)):
            reference = {"sent": [normalize_answer(text) for text in all_tokens["ref.{}".format(i)]]}
            generated = {"sent": [normalize_answer(text) for text in all_tokens["gen.{}".format(i)]]}
            bleu.append(Bleu(4).compute_score(reference, generated)[0][-1])
        return np.mean(bleu)

    def save_predictions(self, predictions):
        assert len(predictions)==len(self), (len(predictions), len(self))
        save_path = os.path.join(self.args.output_dir, "{}{}_predictions.json".format(
            self.data_type if self.args.prefix is None else self.args.prefix,
            "_20200201" if self.args.wiki_2020 and not self.args.ambigqa else ""))
        with open(save_path, "w") as f:
            json.dump(predictions, f)
        self.logger.info("Saved prediction in {}".format(save_path))

class AmbigQGData(AmbigQAData, QAData):
    def __init__(self, logger, args, data_path, is_training, passages=None):
        super(AmbigQGData, self).__init__(logger, args, data_path, is_training, passages)

        with open("/".join(data_path.split("/")[:-2]) + "/nqopen/{}.json".format(self.data_type), "r") as f:
            orig_data = json.load(f)
            id_to_orig_idx = {d["id"]:i for i, d in enumerate(orig_data)}

        self.ref_questions = []
        self.ref_answers = []
        # we will only consider questions with multiple answers
        for i, d in enumerate(self.data):
            if not all([ann["type"]=="multipleQAs" for ann in d["annotations"]]):
                self.ref_questions.append(None)
                self.ref_answers.append(None)
                continue
            questions, answers = [], []
            for annotation in d["annotations"]:
                questions.append([[q.strip() for q in pair["question"].split("|")] for pair in annotation["qaPairs"]])
                answers.append([list(set(pair["answer"])) for pair in annotation["qaPairs"]])
            assert type(answers)==list and \
                all([type(answer)==list for answer in answers]) and \
                all([type(_a)==str for answer in answers+questions for _answer in answer for _a in _answer])
            self.ref_questions.append(questions)
            self.ref_answers.append(answers)
            self.data[i]["orig_idx"] = id_to_orig_idx[d["id"]]


        self.SEP = "<SEP>"
        self.qg_tokenizer = PTBTokenizer()
        self.metric = "EDIT-F1"
        if not self.is_training:
            self.qg_tokenizer = PTBTokenizer()

    # override
    def load_dpr_data(self):
        dpr_retrieval_path = "out/dpr/{}_predictions.json".format(
            self.data_type+"_20200201" if self.args.wiki_2020 else self.data_type)
        postfix = self.tokenizer.__class__.__name__.replace("zer", "zed")
        assert "Bart" in postfix
        dpr_tokenized_path = dpr_retrieval_path.replace("predictions.json", "ambigqa_predictions_{}_qg.json".format(postfix))
        self.load_dpr_data_bart(dpr_retrieval_path, dpr_tokenized_path)

        # in attention_mask, 1 means answer + passages, 2 means prompt, 3 means other answers
        do_include_prompt=True
        do_include_others=True
        new_input_ids, new_attention_mask = [], []
        for input_ids, attention_mask in zip(self.tokenized_data[0], self.tokenized_data[1]):
            _input_ids = [_id for _id, mask in zip(input_ids, attention_mask)
                          if mask==1 or (do_include_prompt and mask==2) or (do_include_others and mask==3)]
            _attention_mask = [1 for mask in attention_mask
                          if mask==1 or (do_include_prompt and mask==2) or (do_include_others and mask==3)]
            assert len(_input_ids)==len(_attention_mask)
            while len(_input_ids)<1024:
                _input_ids.append(self.tokenizer.pad_token_id)
                _attention_mask.append(0)
            new_input_ids.append(_input_ids[:1024])
            new_attention_mask.append(_attention_mask[:1024])
        self.tokenized_data[0] = new_input_ids
        self.tokenized_data[1] = new_attention_mask


    # override
    def load_dpr_data_bart(self, dpr_retrieval_path, dpr_tokenized_path):

        self.logger.info(dpr_tokenized_path)

        if self.is_training and self.args.consider_order_for_multiple_answers:
            dpr_tokenized_path = dpr_tokenized_path.replace(".json", "_ordered.json")

        if os.path.exists(dpr_tokenized_path):
            self.logger.info("Loading DPR data from {}".format(dpr_tokenized_path))
            with open(dpr_tokenized_path, "r") as f:
                self.tokenized_data = json.load(f)
            return

        import itertools
        self.logger.info("Start processing DPR data from {}".format(dpr_retrieval_path))
        if self.passages.tokenized_data is None:
            self.passages.load_tokenized_data("bart", all=True)

        with open(dpr_retrieval_path.format(self.data_type).replace("train", "train_for_inference"), "r") as f:
            dpr_passages = json.load(f)
        assert self.args.psg_sel_dir is not None

        psg_sel_fn = os.path.join(self.args.psg_sel_dir,
                                      "{}{}_psg_sel.json".format(
                                          self.data_type.replace("train", "train_for_inference"),
                                          "_20200201" if self.args.wiki_2020 else ""))
        self.logger.info("Loading passage selection from DPR reader: {}".format(psg_sel_fn))
        with open(psg_sel_fn, "r") as f:
            fg_passages = json.load(f)
            assert len(fg_passages)==len(dpr_passages)
            dpr_passages = [[psgs[i] for i in fg_psgs] for psgs, fg_psgs in zip(dpr_passages, fg_passages)]

        # added to convert original DPR data to AmbigQA DPR data
        dpr_passages = [dpr_passages[d["orig_idx"]] for d in self.data]

        assert len(dpr_passages)==len(self)
        input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, metadata = self.tokenized_data
        assert len(dpr_passages)==len(input_ids)==len(attention_mask)==len(metadata)
        bos_token_id = self.tokenizer.bos_token_id
        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id
        sep_token_id = self.tokenizer.convert_tokens_to_ids(self.SEP)
        assert type(bos_token_id)==type(eos_token_id)==type(sep_token_id)==int


        def _get_tokenized_answer(idx):
            tokens = decoder_input_ids[idx]
            if 0 in decoder_attention_mask[idx]:
                tokens = tokens[:decoder_attention_mask[idx].index(0)]
            assert tokens[0]==tokens[1]==bos_token_id and tokens[-1]==eos_token_id
            return tokens[2:-1]

        def _included(tokens, curr_input_ids):
            for i in range(len(curr_input_ids)+1):
                if curr_input_ids[i:i+len(tokens)]==tokens:
                    return True
            return False

        new_input_ids, new_attention_mask = [], []
        new_output, new_metadata = [], []
        chosen_list = []
        for idx, (curr_input_ids, curr_attention_mask, dpr_ids) in tqdm(enumerate(
                zip(input_ids, attention_mask, dpr_passages))):
            if self.ref_questions[idx] is None:
                continue

            end_of_question = curr_input_ids.index(self.tokenizer.eos_token_id)+1
            q_input_ids = curr_input_ids[:end_of_question]

            p_input_ids, p_attention_mask = [], []
            dpr_input_ids = [self.passages.tokenized_data["input_ids"][_id] for _id in dpr_ids]
            dpr_attention_mask = [self.passages.tokenized_data["attention_mask"][_id] for _id in dpr_ids]
            offset = 0
            while len(p_input_ids)<1024:
                assert dpr_input_ids[offset][0] == bos_token_id
                assert len(dpr_input_ids[offset])==len(dpr_attention_mask[offset])
                assert np.sum(dpr_attention_mask[offset])==len(dpr_attention_mask[offset])
                p_input_ids += dpr_input_ids[offset][1:]
                p_attention_mask += dpr_attention_mask[offset][1:]

            tokenized_ref_answers_list, is_valid_list, n_missing_list = [], [], []
            for ref_questions, ref_answers, ref_metadata in zip(self.ref_questions[idx],
                                                                self.ref_answers[idx],
                                                                metadata[idx]):
                # ref_metadata: [(0, 1), (1, 4)]
                assert type(ref_metadata[0][0])==int
                assert [len(ref_answer)==end-start for ref_answer, (start, end)
                        in zip(ref_answers, ref_metadata)]
                tokenized_ref_answers = [[_get_tokenized_answer(i) for i in range(*m)] for m in ref_metadata]
                is_valid = [[_included(tokens, p_input_ids) for tokens in _tokens] for _tokens in tokenized_ref_answers]
                n_missing = np.sum([not any(v) for v in is_valid])
                tokenized_ref_answers_list.append(tokenized_ref_answers)
                is_valid_list.append(is_valid)
                n_missing_list.append(n_missing)

            min_n_missing = np.min(n_missing_list)
            annotation_indices = [ann_idx for ann_idx in range(len(n_missing_list))
                                  if n_missing_list[ann_idx]==min_n_missing]

            def _form_data(annotation_idx):
                ref_questions = self.ref_questions[idx][annotation_idx]
                ref_answers = self.ref_answers[idx][annotation_idx]
                tokenized_ref_answers = tokenized_ref_answers_list[annotation_idx]
                assert len(ref_questions)==len(ref_answers)==len(tokenized_ref_answers)==len(is_valid_list[annotation_idx])
                final_ref_questions, final_ref_answers = [], []
                chosen_indices = []
                for (ref_question, ref_answer, tok_ref_answer, is_valid) in \
                        zip(ref_questions, ref_answers, tokenized_ref_answers, is_valid_list[annotation_idx]):
                    assert len(ref_answer)==len(tok_ref_answer)==len(is_valid)
                    chosen_idx = is_valid.index(True) if True in is_valid else 0
                    chosen_indices.append(chosen_idx)
                    final_ref_questions.append(ref_question[0])
                    final_ref_answers.append(tok_ref_answer[chosen_idx])
                for i, final_ref_question in enumerate(final_ref_questions):
                    input_ids = [bos_token_id, bos_token_id] + final_ref_answers[i]
                    attention_mask = [1 for _ in input_ids]
                    input_ids += [sep_token_id] + q_input_ids
                    attention_mask += [2 for _ in range(len(q_input_ids)+1)]
                    for j, answer in enumerate(final_ref_answers):
                        if j==i: continue
                        input_ids += [sep_token_id] + answer
                        attention_mask += [3 for _ in range(len(answer)+1)]
                    input_ids += p_input_ids
                    attention_mask += p_attention_mask
                    assert len(input_ids)==len(attention_mask)
                    new_input_ids.append(input_ids)
                    new_attention_mask.append(attention_mask)
                    new_output.append(final_ref_question)
                return chosen_indices

            start = len(new_output)
            if self.is_training:
                start = len(new_output)
                for annotation_idx in annotation_indices:
                    _form_data(annotation_idx)
            else:
                annotation_idx = annotation_indices[0]
                chosen_indices = _form_data(annotation_idx)
                chosen_list.append({"annotation_idx": annotation_idx,
                                    "answer_idx": chosen_indices})
            assert len(new_output)-start > 0
            new_metadata.append((start, len(new_output)))

        if self.is_training:
            new_output = self.tokenizer.batch_encode_plus(new_output, max_length=32, pad_to_max_length=True)
            new_decoder_input_ids = new_output["input_ids"]
            new_decoder_attention_mask = new_output["attention_mask"]
        else:
            new_decoder_input_ids, new_decoder_attention_mask = None, None

        self.tokenized_data = [new_input_ids, new_attention_mask,
                               new_decoder_input_ids, new_decoder_attention_mask, new_metadata]
        if not self.is_training:
            self.tokenized_data.append(chosen_list)
        with open(dpr_tokenized_path, "w") as f:
            json.dump(self.tokenized_data, f)
        self.logger.info("Finish saving tokenized DPR data")


    def load_dataset(self, tokenizer, do_return=False):
        if self.tokenized_data is None:
            self.load_tokenized_data(tokenizer)
            input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, metadata = self.tokenized_data[:5]
        self.dataset = MySimpleQADatasetForPair(input_ids,
                                                attention_mask,
                                                decoder_input_ids if self.is_training or self.args.nq_answer_as_prefix else None,
                                                decoder_attention_mask if self.is_training or self.args.nq_answer_as_prefix else None,
                                                metadata=metadata,
                                                is_training=self.is_training)
        self.logger.info("Loaded {} examples from {} data".format(len(self.dataset), self.data_type))

        if do_return:
            return self.dataset


    # override
    def evaluate(self, predictions, n_paragraphs=None):
        metadata, chosen_list = self.tokenized_data[-2:]
        assert np.sum([ref_questions is not None for ref_questions in self.ref_questions])==len(metadata)
        assert len(predictions)==metadata[-1][-1] and len(chosen_list)==len(metadata)
        # first, tokenize
        data_to_tokenize = {}
        indices = []
        offset = 0
        for i, (d, ref_questions, ref_answers) in enumerate(zip(self.data,  self.ref_questions, self.ref_answers)):
            if ref_questions is None: continue
            data_to_tokenize["prompt.{}".format(i)] = [{"caption": d["question"]}]
            ann_idx = chosen_list[offset]["annotation_idx"]
            answer_idx = chosen_list[offset]["answer_idx"]
            start, end = metadata[offset]
            assert len(ref_questions[ann_idx])==len(ref_answers[ann_idx])==len(answer_idx)==end-start
            indices.append((i, len(answer_idx)))
            for j, (ref_question, pred, a_idx) in enumerate(
                    zip(ref_questions[ann_idx], predictions[start:end], answer_idx)):
                assert type(ref_question)==list
                data_to_tokenize["gen.{}.{}".format(i, j)] = [{"caption": pred if type(pred)==str else pred[0]}]
                data_to_tokenize["ref.{}.{}".format(i, j)] = [{"caption": ref} for ref in ref_question]
            offset += 1

        assert offset==len(metadata)
        all_tokens = self.qg_tokenizer.tokenize(data_to_tokenize)

        def _get(key):
            return {'sent': [normalize_answer(value) for value in all_tokens[key]]}

        bleu, f1s = [], []
        def _get_qg_metrics(gens, refs, prompt, metrics):
            return np.mean([get_qg_metrics(gen, ref, prompt, metrics) for gen, ref in zip(gens, refs)])

        for (i, n) in indices:
            curr_bleu, curr_f1s = [], []
            for j in range(n):
                e = get_qg_metrics(_get("gen.{}.{}".format(i, j)),
                                   _get("ref.{}.{}".format(i, j)),
                                   _get("prompt.{}".format(i)),
                                   metrics=["bleu4", "edit-f1"])
                curr_bleu.append(e["bleu4"])
                curr_f1s.append(e["edit-f1"])
            bleu.append(np.mean(curr_bleu))
            f1s.append(np.mean(curr_f1s))
        self.logger.info("BLEU=%.1f\tEDIT-F1=%.1f" % (100*np.mean(bleu), 100*np.mean(f1s)))
        return np.mean(f1s)


