import os
import torch
import json
import faiss
import numpy as np

from transformers import BertTokenizerFast as BertTokenizer, BertConfig
from models.span_predictor import SpanPredictor
from models.biencoder import MyBiEncoder
from util import decode_span

BASE_DIR = "/data/sewon/dpr"

class InteractiveDPR(object):
    def __init__(self, k=100):
        self.k = k

        wikipedia_path = os.path.join(BASE_DIR, "data/wikipedia_split/filtered_db_text_3_40_30_first.json")
        with open(wikipedia_path, "r") as f:
            self.wikipedia_data = json.load(f)

        postfix = "HNSWFlatIP.store_n=400.ef_contruction=400"
        index_path = os.path.join(BASE_DIR, "checkpoint/retriever/bert-base-encoder-3_40_30_first.Index"+postfix)
        retrieval_checkpoint = os.path.join(BASE_DIR, "checkpoint/retriever/bert-base-encoder.cp")
        reader_checkpoint = os.path.join(BASE_DIR, "checkpoint/reader/nq-bert-base-uncased-32-32-0/best-model.pt")

        def _load_from_checkpoint(Model, checkpoint):
            def convert_to_single_gpu(state_dict):
                if "model_dict" in state_dict:
                    state_dict = state_dict["model_dict"]
                def _convert(key):
                    if key.startswith('module.'):
                        return key[7:]
                    return key
                return {_convert(key):value for key, value in state_dict.items()}
            state_dict = convert_to_single_gpu(torch.load(checkpoint))
            model = Model(BertConfig.from_pretrained("bert-base-uncased"))
            return model.from_pretrained(None, config=model.config, state_dict=state_dict)

        self.question_encoder = _load_from_checkpoint(MyBiEncoder, retrieval_checkpoint).question_model.cuda()
        self.reader = _load_from_checkpoint(SpanPredictor, reader_checkpoint).cuda()
        self.question_encoder.eval()
        self.reader.eval()

        self.index = faiss.read_index(index_path)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        print ("Ready for a demo!")

    def run(self, questions, max_length=160, topk_answer=10):
        is_single = type(questions)==str
        if is_single:
            questions = [questions]
        questions = [question[:-1] if question.endswith("?") else question for question in questions]
        question_input = self.tokenizer.batch_encode_plus(questions,
                                                          padding="max_length",
                                                          max_length=32)
        passages, retrieval_scores = self.retrieve(question_input)
        input_ids, attention_mask, token_type_ids, offsets_mapping = [], [], [], []
        for question, q_input, passage in zip(questions, question_input, passages):
            question = question if question.endswith("?") else question+"?"
            input_data = self.tokenizer.batch_encode_plus(
                [(question, self.wikipedia_data[idx][0] + " " + self.tokenizer.sep_token + " "
                  + self.wikipedia_data[idx][1]) for idx in passage],
                padding="max_length", max_length=max_length,
                return_offsets_mapping=True, truncation=True)
            input_ids.append(input_data["input_ids"])
            attention_mask.append(input_data["attention_mask"])
            token_type_ids.append(input_data["token_type_ids"])
            offsets_mapping.append(input_data["offset_mapping"])

        assert len(input_ids)==len(attention_mask)==len(token_type_ids)==len(questions)==len(offsets_mapping)
        assert np.all([len(l)==self.k and np.all([len(li)==max_length for li in l])
                       for l in input_ids+attention_mask+token_type_ids])

        start_logits, end_logits, sel_logits = self.read(input_ids, attention_mask, token_type_ids, topk_answer)
        outputs = []
        for _input_ids, _attention_mask, _start_logits, _end_logits, _sel_logits, offset_mapping in zip(
                input_ids, attention_mask, start_logits, end_logits, sel_logits, offsets_mapping):
            output = decode_span((_input_ids, _attention_mask),
                                 self.tokenizer, _start_logits, _end_logits, _sel_logits,
                                 max_answer_length=10, topk_answer=topk_answer)
            curr_output = []
            for j, o in enumerate(output):
                passage_id = passage[o["passage_index"]]
                title, text = self.wikipedia_data[passage_id]

                spans = offset_mapping[o["passage_index"]][o["start_index"]+o["start_offset"]:o["end_index"]+1+o["start_offset"]]
                char_start, char_end = spans[0][0]-len(title)-7, spans[-1][-1]-len(title)-7

                curr_output.append({
                    "passage_index": o["passage_index"],
                    "title": title,
                    "passage": text[:char_start] + "<span class='red'><strong>" + text[char_start:char_end] + "</strong></span>" + text[char_end:],
                    "softmax": {"passage": np.exp(o["log_softmax"][0]),
                                "span": np.exp(o["log_softmax"][1]),
                                "joint": np.exp(np.sum(o["log_softmax"]))}})
            outputs.append(curr_output)
        if is_single:
            return outputs[0]
        return outputs

    def retrieve(self, question_input):
        input_ids, attention_mask = [torch.LongTensor(t).cuda() for t in
                                     [question_input["input_ids"], question_input["attention_mask"]]]
        with torch.no_grad():
            query_vectors = self.question_encoder(input_ids, attention_mask).last_hidden_state[:,0,:]
        query_vectors = query_vectors.detach().cpu().numpy()
        aux_dim = np.zeros(len(query_vectors), dtype="float32")
        query_nhsw_vectors = np.hstack((query_vectors, aux_dim.reshape(-1, 1)))
        D, I = self.index.search(query_nhsw_vectors, self.k)
        return I.tolist(), D.tolist()

    def read(self, input_ids, attention_mask, token_type_ids, topk_answer):
        input_ids_pt, attention_mask_pt, token_type_ids_pt = \
            [torch.LongTensor(t).cuda() for t in [input_ids, attention_mask, token_type_ids]]
        with torch.no_grad():
            start_logits, end_logits, sel_logits = self.reader(input_ids_pt, attention_mask_pt, token_type_ids_pt)

        start_logits, end_logits, sel_logits = [l.detach().cpu().numpy().tolist() for l in [start_logits, end_logits, sel_logits]]
        return start_logits, end_logits, sel_logits

if __name__=='__main__':
    dpr = InteractiveDPR()

    with open("data/nqopen-dev.json", "r") as f:
        data = json.load(f)
    output = dpr.run(data[0]["question"])
    from IPython import embed; embed()






