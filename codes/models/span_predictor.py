import torch
import  numpy
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import BertForQuestionAnswering, AlbertForQuestionAnswering

class SpanPredictor(BertForQuestionAnswering):
    def __init__(self, config):
        config.num_labels = 2
        super().__init__(config)
        self.qa_classifier = nn.Linear(config.hidden_size, 1)

    def forward(self,
                input_ids=None, attention_mask=None,
                token_type_ids=None, inputs_embeds=None,
                start_positions=None, end_positions=None, answer_mask=None,
                is_training=False):

        N, M, L = input_ids.size()
        output = self.bert(input_ids.view(N*M, L),
                           attention_mask=attention_mask.view(N*M, L),
                           token_type_ids=token_type_ids.view(N*M, L),
                           inputs_embeds=None if inputs_embeds is None else inputs_embeds.view(N*M, L, -1))[0]
        logits = self.qa_outputs(output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        sel_logits = self.qa_classifier(output[:,0,:])

        if is_training:
            start_positions, end_positions, answer_mask = \
                start_positions.view(N*M, -1), end_positions.view(N*M, -1), answer_mask.view(N*M, -1)
            return get_loss(start_positions, end_positions, answer_mask,
                            start_logits, end_logits, sel_logits, N, M)
        else:
            return start_logits.view(N, M, L), end_logits.view(N, M, L), sel_logits.view(N, M)


class AlbertSpanPredictor(AlbertForQuestionAnswering):
    def __init__(self, config):
        config.num_labels = 2
        super().__init__(config)
        self.qa_classifier = nn.Linear(config.hidden_size, 1)

    def forward(self,
                input_ids=None, attention_mask=None,
                token_type_ids=None, inputs_embeds=None,
                start_positions=None, end_positions=None, answer_mask=None,
                is_training=False):

        N, M, L = input_ids.size()
        output = self.albert(input_ids.view(N*M, L),
                             attention_mask=attention_mask.view(N*M, L),
                             token_type_ids=token_type_ids.view(N*M, L),
                             inputs_embeds=None if inputs_embeds is None else inputs_embeds.view(N*M, L, -1))[0]
        logits = self.qa_outputs(output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        sel_logits = self.qa_classifier(output[:,0,:])

        if is_training:
            start_positions, end_positions, answer_mask = \
                start_positions.view(N*M, -1), end_positions.view(N*M, -1), answer_mask.view(N*M, -1)
            return get_loss(start_positions, end_positions, answer_mask,
                            start_logits, end_logits, sel_logits, N, M)
        else:
            return start_logits.view(N, M, L), end_logits.view(N, M, L), sel_logits.view(N, M)

def get_loss(start_positions, end_positions, answer_mask, start_logits, end_logits, sel_logits, N, M):
    answer_mask = answer_mask.type(torch.FloatTensor).cuda()
    ignored_index = start_logits.size(1)
    start_positions.clamp_(0, ignored_index)
    end_positions.clamp_(0, ignored_index)
    loss_fct = CrossEntropyLoss(reduce=False, ignore_index=ignored_index)

    sel_logits = sel_logits.view(N, M)
    sel_labels = torch.zeros(N, dtype=torch.long).cuda()
    sel_loss = torch.sum(loss_fct(sel_logits, sel_labels))
    start_losses = [(loss_fct(start_logits, _start_positions) * _span_mask) \
                    for (_start_positions, _span_mask) \
                    in zip(torch.unbind(start_positions, dim=1), torch.unbind(answer_mask, dim=1))]
    end_losses = [(loss_fct(end_logits, _end_positions) * _span_mask) \
                    for (_end_positions, _span_mask) \
                    in zip(torch.unbind(end_positions, dim=1), torch.unbind(answer_mask, dim=1))]
    loss_tensor = torch.cat([t.unsqueeze(1) for t in start_losses], dim=1) + \
        torch.cat([t.unsqueeze(1) for t in end_losses], dim=1)
    loss_tensor=loss_tensor.view(N, M, -1).max(dim=1)[0]
    span_loss = _take_mml(loss_tensor)
    return span_loss + sel_loss

def _take_mml(loss_tensor):
    marginal_likelihood = torch.sum(torch.exp(
            - loss_tensor - 1e10 * (loss_tensor==0).float()), 1)
    return -torch.sum(torch.log(marginal_likelihood + \
                                torch.ones(loss_tensor.size(0)).cuda()*(marginal_likelihood==0).float()))

