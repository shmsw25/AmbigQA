import numpy as np
import time
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler

class MySimpleQADataset(Dataset):
    def __init__(self,
                 input_ids, attention_mask,
                 decoder_input_ids=None, decoder_attention_mask=None,
                 in_metadata=None, out_metadata=None,
                 is_training=False,
                 answer_as_prefix=False):
        self.input_ids = torch.LongTensor(input_ids)
        self.attention_mask = torch.LongTensor(attention_mask)
        self.decoder_input_ids = None if decoder_input_ids is None else torch.LongTensor(decoder_input_ids)
        self.decoder_attention_mask = None if decoder_attention_mask is None else torch.LongTensor(decoder_attention_mask)
        self.in_metadata = list(zip(range(len(input_ids)), range(1, 1+len(input_ids)))) \
            if in_metadata is None else in_metadata
        self.out_metadata = list(zip(range(len(decoder_input_ids)), range(1, 1+len(decoder_input_ids)))) \
            if is_training and out_metadata is None else out_metadata
        self.is_training = is_training
        self.answer_as_prefix = answer_as_prefix

        assert len(self.input_ids)==len(self.attention_mask)==self.in_metadata[-1][-1], \
            (len(self.input_ids), len(self.attention_mask), self.in_metadata[-1])
        assert not self.is_training or len(self.decoder_input_ids)==len(self.decoder_attention_mask)==self.out_metadata[-1][-1]

    def __len__(self):
        return len(self.in_metadata)

    def __getitem__(self, idx):
        if not self.is_training:
            idx = self.in_metadata[idx][0]
            if self.answer_as_prefix:
                #assert self.out_metadata[idx][1]-self.out_metadata[idx][0]==1
                out_idx = self.out_metadata[idx][0]
                return self.input_ids[idx], self.attention_mask[idx], \
                    self.decoder_input_ids[out_idx], self.decoder_attention_mask[out_idx]
            return self.input_ids[idx], self.attention_mask[idx]

        in_idx = np.random.choice(range(*self.in_metadata[idx]))
        out_idx = np.random.choice(range(*self.out_metadata[idx]))
        return self.input_ids[in_idx], self.attention_mask[in_idx], \
            self.decoder_input_ids[out_idx], self.decoder_attention_mask[out_idx]

class MySimpleQADatasetForPair(Dataset):
    def __init__(self,
                 input_ids, attention_mask,
                 decoder_input_ids=None, decoder_attention_mask=None, metadata=None,
                 is_training=False):
        self.input_ids = torch.LongTensor(input_ids)
        self.attention_mask = torch.LongTensor(attention_mask)
        self.decoder_input_ids = None if decoder_input_ids is None else torch.LongTensor(decoder_input_ids)
        self.decoder_attention_mask = None if decoder_attention_mask is None else torch.LongTensor(decoder_attention_mask)
        self.metadata = metadata
        self.is_training = is_training

        assert len(self.input_ids)==len(self.attention_mask)
        assert not self.is_training or len(self.input_ids)==len(self.decoder_input_ids)==len(self.decoder_attention_mask)==self.metadata[-1][-1]
        assert self.metadata[-1][-1]==len(self.input_ids)

    def __len__(self):
        return len(self.metadata) if self.is_training else len(self.input_ids)

    def __getitem__(self, idx):
        if not self.is_training:
            return self.input_ids[idx], self.attention_mask[idx]
        idx = np.random.choice(range(*self.metadata[idx]))
        return self.input_ids[idx], self.attention_mask[idx], \
            self.decoder_input_ids[idx], self.decoder_attention_mask[idx]

class MyQADataset(Dataset):
    def __init__(self, data,
                 is_training=False, train_M=None, test_M=None):
        self.data = data #.dictionify()
        self.positive_input_ids = self.tensorize("positive_input_ids")
        self.positive_input_mask = self.tensorize("positive_input_mask")
        self.positive_token_type_ids = self.tensorize("positive_token_type_ids")
        assert len(self.positive_input_ids)==len(self.positive_input_mask)==len(self.positive_token_type_ids)

        if is_training:
            self.positive_start_positions = self.tensorize("positive_start_positions")
            self.positive_end_positions = self.tensorize("positive_end_positions")
            self.positive_answer_mask = self.tensorize("positive_answer_mask")
            self.negative_input_ids = self.tensorize("negative_input_ids")
            self.negative_input_mask = self.tensorize("negative_input_mask")
            self.negative_token_type_ids = self.tensorize("negative_token_type_ids")
            assert len(self.negative_input_ids)==len(self.negative_input_mask)==len(self.negative_token_type_ids)
            assert len(self.positive_input_ids)==\
                    len(self.positive_start_positions)==len(self.positive_end_positions)==len(self.positive_answer_mask)
            assert all([len(positive_input_ids)>0 for positive_input_ids in self.positive_input_ids])

        self.is_training = is_training
        self.train_M = train_M
        self.test_M = test_M

    def __len__(self):
        return len(self.positive_input_ids)

    def __getitem__(self, idx):
        if not self.is_training:
            input_ids = self.positive_input_ids[idx][:self.test_M]
            input_mask = self.positive_input_mask[idx][:self.test_M]
            token_type_ids = self.positive_token_type_ids[idx][:self.test_M]
            return [self._pad(t, self.test_M) for t in [input_ids, input_mask, token_type_ids]]

        # sample positive
        positive_idx = np.random.choice(len(self.positive_input_ids[idx]))
        #positive_idx = 0
        positive_input_ids = self.positive_input_ids[idx][positive_idx]
        positive_input_mask = self.positive_input_mask[idx][positive_idx]
        positive_token_type_ids = self.positive_token_type_ids[idx][positive_idx]
        positive_start_positions = self.positive_start_positions[idx][positive_idx]
        positive_end_positions = self.positive_end_positions[idx][positive_idx]
        positive_answer_mask = self.positive_answer_mask[idx][positive_idx]

        # sample negatives
        negative_idxs = np.random.permutation(range(len(self.negative_input_ids[idx])))[:self.train_M-1]
        negative_input_ids = [self.negative_input_ids[idx][i] for i in negative_idxs]
        negative_input_mask = [self.negative_input_mask[idx][i] for i in negative_idxs]
        negative_token_type_ids = [self.negative_token_type_ids[idx][i] for i in negative_idxs]
        negative_input_ids, negative_input_mask, negative_token_type_ids = \
            [self._pad(t, self.train_M-1) for t in [negative_input_ids, negative_input_mask, negative_token_type_ids]]

        # aggregate
        input_ids = torch.cat([positive_input_ids.unsqueeze(0), negative_input_ids], dim=0)
        input_mask = torch.cat([positive_input_mask.unsqueeze(0), negative_input_mask], dim=0)
        token_type_ids = torch.cat([positive_token_type_ids.unsqueeze(0), negative_token_type_ids], dim=0)
        start_positions, end_positions, answer_mask = \
            [self._pad([t], self.train_M) for t in [positive_start_positions,
                                                  positive_end_positions,
                                                  positive_answer_mask]]
        return input_ids, input_mask, token_type_ids, start_positions, end_positions, answer_mask

    def tensorize(self, key):
        return [torch.LongTensor(t) for t in self.data[key]] if key in self.data.keys() else None

    def _pad(self, input_ids, M):
        if len(input_ids)==0:
            return torch.zeros((M, self.negative_input_ids[0].size(1)), dtype=torch.long)
        if type(input_ids)==list:
            input_ids = torch.stack(input_ids)
        if len(input_ids)==M:
            return input_ids
        return torch.cat([input_ids,
                          torch.zeros((M-input_ids.size(0), input_ids.size(1)), dtype=torch.long)],
                         dim=0)

class MyDataLoader(DataLoader):

    def __init__(self, args, dataset, is_training, batch_size=None):
        if is_training:
            sampler=RandomSampler(dataset)
            batch_size = args.train_batch_size if batch_size is None else batch_size
        else:
            sampler=SequentialSampler(dataset)
            batch_size = args.predict_batch_size if batch_size is None else batch_size

        super(MyDataLoader, self).__init__(dataset, sampler=sampler, batch_size=batch_size)


