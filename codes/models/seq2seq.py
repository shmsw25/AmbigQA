import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import T5ForConditionalGeneration, BartForConditionalGeneration

class MyBart(BartForConditionalGeneration):
    def forward(self, input_ids, attention_mask=None, encoder_outputs=None,
            decoder_input_ids=None, decoder_attention_mask=None, decoder_cached_states=None,
            use_cache=False, is_training=False):

        if is_training:
            decoder_start_token_id = self.config.decoder_start_token_id
            _decoder_input_ids = decoder_input_ids.new_zeros(decoder_input_ids.shape)
            _decoder_input_ids[..., 1:] = decoder_input_ids[..., :-1].clone()
            _decoder_input_ids[..., 0] = decoder_start_token_id
            #_decoder_input_ids = decoder_input_ids.clone()
            #_decoder_input_ids[..., 0] = decoder_start_token_id
            #new_decoder_input_ids = decoder_input_ids.new_zeros(decoder_input_ids.shape)
            #new_decoder_input_ids[..., :-1] = decoder_input_ids[..., 1:].clone()
            #decoder_input_ids = new_decoder_input_ids
            #print (input_ids[0,:10].detach().cpu().tolist())
            #print (_decoder_input_ids[0,:10].detach().cpu().tolist())
            #print (decoder_input_ids[0, :10].detach().cpu().tolist())
        else:
            _decoder_input_ids = decoder_input_ids.clone()
            #print (input_ids[0,:10].detach().cpu().tolist())
            #print (_decoder_input_ids[0].detach().cpu().tolist())

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=_decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            decoder_cached_states=decoder_cached_states,
            use_cache=use_cache,
        )
        lm_logits = F.linear(outputs[0], self.model.shared.weight, bias=self.final_logits_bias)
        if is_training:
            loss_fct = nn.CrossEntropyLoss(reduce=False)
            losses = loss_fct(lm_logits.view(-1, self.config.vocab_size),
                              decoder_input_ids.view(-1))
            loss = torch.sum(losses * decoder_attention_mask.float().view(-1))
            return loss
        return (lm_logits, ) + outputs[1:]

class MyT5(T5ForConditionalGeneration):
    def forward(self, input_ids=None, attention_mask=None, encoder_outputs=None,
            decoder_input_ids=None, decoder_attention_mask=None, decoder_cached_states=None,
            decoder_past_key_value_states=None,
            use_cache=False, is_training=False):

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids, attention_mask=attention_mask,
                inputs_embeds=None, head_mask=None
            )
        hidden_states = encoder_outputs[0]

        _decoder_input_ids = decoder_input_ids
        _decoder_attention_mask = decoder_attention_mask
        if is_training:
            decoder_start_token_id = self.config.decoder_start_token_id
            _decoder_input_ids = decoder_input_ids.new_zeros(decoder_input_ids.shape)
            _decoder_input_ids[..., 1:] = decoder_input_ids[..., :-1].clone()
            _decoder_input_ids = _decoder_input_ids + self.config.eos_token_id * (_decoder_input_ids==0).long()
            _decoder_input_ids[..., 0] = decoder_start_token_id
            _decoder_attention_mask = decoder_attention_mask.new_zeros(decoder_attention_mask.shape)
            _decoder_attention_mask[..., 1:] = decoder_attention_mask[..., :-1].clone()
            _decoder_attention_mask[..., 0] = 1
        else:
            print (_decoder_input_ids)
            print (_decoder_attention_mask)
        decoder_outputs = self.decoder(
            input_ids=_decoder_input_ids,
            attention_mask=_decoder_attention_mask,
            inputs_embeds=None,
            past_key_value_states=None,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=None,
            use_cache=use_cache,
        )

        sequence_output = decoder_outputs[0]
        sequence_output = sequence_output * (self.model_dim ** -0.5)
        lm_logits = self.lm_head(sequence_output)

        decoder_outputs = (lm_logits,) + decoder_outputs[1:]  # Add hidden states and attention if they are here
        if is_training:
            loss_fct = nn.CrossEntropyLoss(reduce=False)
            losses = loss_fct(lm_logits.view(-1, self.config.vocab_size),
                              decoder_input_ids.view(-1))
            loss = torch.sum(losses * decoder_attention_mask.float().view(-1))
            return loss

        return decoder_outputs + encoder_outputs


