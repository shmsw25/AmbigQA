import math
import numpy  as np
from tqdm import tqdm

from joblib import Parallel, delayed

def decode_span_batch(features, scores, tokenizer, max_answer_length,
                      n_paragraphs=None, topk_answer=1, verbose=False, n_jobs=1,
                      save_psg_sel_only=False):
    assert len(features)==len(scores)
    iter=zip(features, scores)
    if n_jobs>1:
        def f(t):
            return decode_span(t[0], tokenizer, t[1][0], t[1][1], t[1][2], max_answer_length,
                               n_paragraphs=n_paragraphs, topk_answer=topk_answer,
                               save_psg_sel_only=save_psg_sel_only)
        return Parallel(n_jobs=n_jobs)(delayed(f)(t) for t in iter)
    if verbose:
        iter = tqdm(iter)
    predictions = [decode_span(feature, tokenizer, start_logits, end_logits, sel_logits,
                        max_answer_length, n_paragraphs, topk_answer, save_psg_sel_only) \
            for (feature, (start_logits, end_logits, sel_logits)) in iter]
    return predictions

def decode_span(feature, tokenizer, start_logits_list, end_logits_list, sel_logits_list,
                max_answer_length, n_paragraphs=None, topk_answer=1, save_psg_sel_only=False):
    all_positive_token_ids, all_positive_input_mask = feature
    assert len(start_logits_list)==len(end_logits_list)==len(sel_logits_list)
    assert type(sel_logits_list[0])==float
    log_softmax_switch_logits_list = _compute_log_softmax(sel_logits_list[:len(all_positive_token_ids)])

    if save_psg_sel_only:
        return np.argsort(-np.array(log_softmax_switch_logits_list)).tolist()

    sorted_logits = sorted(enumerate(zip(start_logits_list, end_logits_list, sel_logits_list)),
                           key=lambda x: -x[1][2])
    nbest = []
    for passage_index, (start_logits, end_logits, switch_logits) in sorted_logits:
        scores = []
        if len(all_positive_token_ids)<=passage_index:
            continue

        positive_token_ids = all_positive_token_ids[passage_index]
        positive_input_mask = all_positive_input_mask[passage_index]
        start_offset = 1 + positive_token_ids.index(tokenizer.sep_token_id)
        end_offset = positive_input_mask.index(0) if 0 in positive_input_mask else len(positive_input_mask)

        positive_token_ids = positive_token_ids[start_offset:end_offset]
        start_logits = start_logits[start_offset:end_offset]
        end_logits = end_logits[start_offset:end_offset]
        log_softmax_start_logits = _compute_log_softmax(start_logits)
        log_softmax_end_logits = _compute_log_softmax(end_logits)

        for (i, s) in enumerate(start_logits):
            for (j, e) in enumerate(end_logits[i:i+max_answer_length]):
                scores.append(((i, i+j), s+e))

        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        chosen_span_intervals = []

        for (start_index, end_index), score in scores:
            if end_index < start_index:
                continue
            length = end_index - start_index + 1
            if length > max_answer_length:
                continue
            if any([start_index<=prev_start_index<=prev_end_index<=end_index or
                    prev_start_index<=start_index<=end_index<=prev_end_index
                    for (prev_start_index, prev_end_index) in chosen_span_intervals]):
                continue

            answer_text = tokenizer.decode(positive_token_ids[start_index:end_index+1],
                                           skip_special_tokens=True,
                                           clean_up_tokenization_spaces=True).strip()
            passage_text = tokenizer.decode(positive_token_ids[:start_index],
                                            skip_special_tokens=True,
                                            clean_up_tokenization_spaces=True).strip() + \
                " <answer>" + answer_text + "</answer> " + \
                tokenizer.decode(positive_token_ids[end_index+1:],
                                 skip_special_tokens=True,
                                 clean_up_tokenization_spaces=True).strip()

            nbest.append({
                'text': answer_text,
                'passage_index': passage_index,
                'passage': passage_text,
                'log_softmax': log_softmax_switch_logits_list[passage_index] + \
                                log_softmax_start_logits[start_index] + \
                                log_softmax_end_logits[end_index]})

            chosen_span_intervals.append((start_index, end_index))
            if topk_answer>-1 and topk_answer==len(chosen_span_intervals):
                break

    if len(nbest)==0:
        nbest = [{'text': 'empty', 'log_softmax': -99999, 'passage_index': 0, 'passage': ''}]

    sorted_nbest = sorted(nbest, key=lambda x: -x["log_softmax"])

    if n_paragraphs is None:
        return sorted_nbest[:topk_answer] if topk_answer>-1 else sorted_nbest
    else:
        return [[pred for pred in sorted_nbest if pred['passage_index']<n] for n in n_paragraphs]

def _compute_log_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []
    if type(scores[0])==tuple:
        scores = [s[1] for s in scores]
    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score
    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x
    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return np.log(probs).tolist()
