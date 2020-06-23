# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This file implements:
Ghazvininejad, Marjan, et al.
"Constant-time machine translation with conditional masked language models."
arXiv preprint arXiv:1904.09324 (2019).
"""
import torch
import torch.nn.functional as F

from fairseq.models import register_model, register_model_architecture
from fairseq.models.nat import NATDiscoLayerDecoder, NATransformerModel, NATransformerDecoder, ensemble_decoder
from fairseq.utils import new_arange
from fairseq.modules.transformer_sentence_encoder import init_bert_params
import numpy as np

def _skeptical_unmasking(output_scores, output_masks, p):
    sorted_index = output_scores.sort(-1)[1]
    boundary_len = (
        (output_masks.sum(1, keepdim=True).type_as(output_scores) - 2) * p
    ).long()
    skeptical_mask = new_arange(output_masks) < boundary_len
    return skeptical_mask.scatter(1, sorted_index, skeptical_mask)


class NATransformerDiscoDecoder(NATDiscoLayerDecoder):

    def forward_embedding(self, prev_output_tokens, states=None):
        # override forward embedding to separate q and kv input.
        # embed positions
        positions = (
            self.embed_positions(prev_output_tokens)
            if self.embed_positions is not None
            else None
        )

        # embed tokens and positions
        if states is None:
            x = self.embed_scale * self.embed_tokens(prev_output_tokens)
            if self.project_in_dim is not None:
                x = self.project_in_dim(x)
        else:
            x = states

        if positions is not None:
            x += positions
            positions = F.dropout(positions, p=self.dropout, training=self.training)
        x = F.dropout(x, p=self.dropout, training=self.training)
        decoder_padding_mask = prev_output_tokens.eq(self.padding_idx)
        return positions, decoder_padding_mask, x


    @ensemble_decoder
    def forward(self, normalize, encoder_out, prev_output_tokens, step=0, masking_type='random_masking', gen_order=None, **unused):
        q_mask = self.get_qmask(prev_output_tokens, masking_type=masking_type, gen_order=gen_order)
        features, _ = self.extract_features(
            prev_output_tokens,
            q_mask=q_mask,
            encoder_out=encoder_out,
            embedding_copy=(step == 0) & self.src_embedding_copy,
            masking_type=masking_type,
        )
        decoder_out = self.output_layer(features)
        return F.log_softmax(decoder_out, -1) if normalize else decoder_out

    def extract_features(
        self,
        prev_output_tokens,
        q_mask,
        encoder_out=None,
        early_exit=None,
        embedding_copy=False,
        masking_type='random_masking',
        **unused
    ):
        """
        Similar to *forward* but only return features.

        Inputs:
            prev_output_tokens: Tensor(B, T)
            encoder_out: a dictionary of hidden states and masks

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
            the LevenshteinTransformer decoder has full-attention to all generated tokens
        """
        # embedding
        if embedding_copy:
            src_embd = encoder_out.encoder_embedding
            src_mask = encoder_out.encoder_padding_mask
            src_mask = (
                ~src_mask
                if src_mask is not None
                else prev_output_tokens.new_ones(*src_embd.size()[:2]).bool()
            )

            x, decoder_padding_mask, x_kv = self.forward_embedding(
                prev_output_tokens,
                self.forward_copying_source(
                    src_embd, src_mask, prev_output_tokens.ne(self.padding_idx)
                ),
            )

        else:

            x, decoder_padding_mask, x_kv = self.forward_embedding(prev_output_tokens)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        x_kv = x_kv.transpose(0, 1)
        attn = None
        inner_states = [x]

        # decoder layers
        for i, layer in enumerate(self.layers):

            # early exit from the decoder.
            if (early_exit is not None) and (i >= early_exit):
                break

            x, attn, _ = layer(
                x,
                x_kv,
                encoder_out.encoder_out if encoder_out is not None else None,
                encoder_out.encoder_padding_mask if encoder_out is not None else None,
                self_attn_mask=None,
                self_attn_padding_mask=decoder_padding_mask,
                q_mask = q_mask,
            )
            inner_states.append(x)

        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": attn, "inner_states": inner_states}

    def get_qmask(self, prev_output_tokens, masking_type, gen_order=None):
        # prev_output_tokens [B, T]
        # decoder_padding_mask [B, T]
        # return [B, T, T]
        # We softmax over the last dimension
        # Always attend to eos to avoid the all negative inf edge case.
        # Namely, [:, :, eos_idxes] = False (never mask out the attention to eos)
        decoder_padding_mask = prev_output_tokens.eq(self.padding_idx)
        bsz, max_len = decoder_padding_mask.shape
        if masking_type=='token_masking':
            q_mask = prev_output_tokens.eq(self.unk).unsqueeze(1).repeat([1, max_len, 1])
            return q_mask

        elif masking_type=='full_masking':
            q_mask = prev_output_tokens.float().new_ones([bsz, max_len, max_len])
            q_mask = q_mask.bool()
            ## EOS is always available
            q_mask = q_mask.masked_fill_(prev_output_tokens.unsqueeze(1).eq(self.bos), False)
            q_mask = q_mask.masked_fill_(prev_output_tokens.unsqueeze(1).eq(self.eos), False)
            return q_mask

        elif masking_type=='easy_first_masking':
            assert gen_order is not None
            # mask out yourself and later tokens
            ## EOS, BOS, and pad do not see any other token.
            q_mask = gen_order.unsqueeze(2) <= gen_order.unsqueeze(1)
            q_mask = q_mask.masked_fill_(prev_output_tokens.unsqueeze(2).eq(self.eos), True)
            q_mask = q_mask.masked_fill_(prev_output_tokens.unsqueeze(2).eq(self.bos), True)
            q_mask = q_mask.masked_fill_(prev_output_tokens.unsqueeze(2).eq(self.padding_idx), True)
            ## EOS is always available
            q_mask = q_mask.masked_fill_(prev_output_tokens.unsqueeze(1).eq(self.bos), False)
            q_mask = q_mask.masked_fill_(prev_output_tokens.unsqueeze(1).eq(self.eos), False)
            q_mask = q_mask.masked_fill_(prev_output_tokens.unsqueeze(1).eq(self.padding_idx), True)
            return q_mask

        assert masking_type=='random_masking'
        # never attend to yourself. You are predicting yourself. C.f. vanilla autoregressive
        q_mask = prev_output_tokens.float().new_zeros([bsz, max_len, max_len])
        # [bsz, max_len, max_len]
        # First generate uniform (0, 1) to determine which words to mask randomly
        if not self.training:
            # evaluation model. Use numpy seed.
            random_score = torch.Tensor(self.random.uniform(size = q_mask.shape)).to(q_mask.device)
            cutoff_ratio = torch.Tensor(self.random.uniform(size = [bsz, max_len])).to(q_mask.device)
        else:
            seed = 0
            self.random = np.random.RandomState(seed)
            random_score = q_mask.uniform_()
            cutoff_ratio = q_mask.new_zeros([bsz, max_len]).uniform_()
        ## bos, eos and pad cannot see anyone so no information leakage.
        random_score = random_score.masked_fill_(prev_output_tokens.unsqueeze(2).eq(self.bos), 5.0)
        random_score = random_score.masked_fill_(prev_output_tokens.unsqueeze(2).eq(self.eos), 5.0)
        random_score = random_score.masked_fill_(prev_output_tokens.unsqueeze(2).eq(self.padding_idx), 5.0)
        # Always mask pads. Put 5.0 so you always mask them.
        random_score = random_score.masked_fill_(decoder_padding_mask.unsqueeze(1), 5.0)
        # Always mask yourself. Put 5.0 so you always mask it.
        random_score = random_score.masked_fill_(torch.diag(q_mask.new_ones([max_len])).bool().unsqueeze(0), 5.0)
        ## We always unmask eos. So set them -5.0.
        random_score = random_score.masked_fill_(prev_output_tokens.unsqueeze(1).eq(self.bos), -5.0)
        random_score = random_score.masked_fill_(prev_output_tokens.unsqueeze(1).eq(self.eos), -10.0)
        sorted_random_score, target_ordering = random_score.sort(2)
        _, target_rank = target_ordering.sort(2)
        target_lengths = max_len - decoder_padding_mask.float().sum(1, keepdim=True) - 2
        # -2 for bos and eos.
        # target_lengths: [bsz, 1]
        cutoff_len = (target_lengths * cutoff_ratio).long() + 2
        # Cutoff_len chooses the number of unmasked words from [2, seq_length-1], including bos and eos
        # Hardest case: we unmask only eos [1]
        # Easiest case: we unmask all but yourself [seq_length-1]
        q_mask = target_rank < cutoff_len.unsqueeze(2)
        # q_mask should be swapped. True for tokens that you DO NOT attend to.
        q_mask = ~q_mask
        return q_mask

    


@register_model("disco_transformer")
class DiscoTransformerModel(NATransformerModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        # No predetermined noise for disco training. On-the-fly masking.
        #assert self.args.noise == 'no_noise'

    def forward(
        self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, q_mask=None, masking_type='random_masking',  **kwargs
    ):
        assert not self.decoder.src_embedding_copy, "do not support embedding copy."

        # encoding
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        # length prediction
        length_out = self.decoder.forward_length(normalize=False, encoder_out=encoder_out)
        length_tgt = self.decoder.forward_length_prediction(length_out, encoder_out, tgt_tokens)

        # decoding
        word_ins_out = self.decoder(
            normalize=False,
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out,
            masking_type=masking_type)
        word_ins_mask = prev_output_tokens.eq(self.unk)
        word_ins_mask = prev_output_tokens.ne(self.pad) & prev_output_tokens.ne(self.bos) & prev_output_tokens.ne(self.eos)
        nsents = word_ins_mask.size(0)
        ntokens = float(word_ins_mask.sum())

        return {
            "word_ins": {
                "out": word_ins_out, "tgt": tgt_tokens,
                "mask": word_ins_mask, "ls": self.args.label_smoothing,
                "nll_loss": True
            },
            "length": {
                "out": length_out, "tgt": length_tgt,
                "factor": nsents/ntokens
                #"factor": self.decoder.length_loss_factor
            }
        }

    def forward_decoder(self, decoder_out, encoder_out, decoding_format=None, **kwargs):

        step = decoder_out.step
        max_step = decoder_out.max_step

        output_tokens = decoder_out.output_tokens
        output_scores = decoder_out.output_scores
        history = decoder_out.history
        if decoding_format == 'mask-predict' or decoding_format is None:
            output_tokens, output_scores = self.mask_predict(output_tokens, output_scores, encoder_out, history, step, max_step)
        else:
            assert decoding_format == 'easy-first'
            output_tokens, output_scores = self.easy_first(output_tokens, output_scores, encoder_out, history, step, max_step)

        return decoder_out._replace(
            output_tokens=output_tokens,
            output_scores=output_scores,
            attn=None,
            history=history
        )

    def mask_predict(self, output_tokens, output_scores, encoder_out, history, step, max_step):
        # execute the decoder
        output_masks = output_tokens.eq(self.unk)
        _scores, _tokens = self.decoder(
            normalize=True,
            prev_output_tokens=output_tokens,
            encoder_out=encoder_out,
            masking_type='token_masking',
        ).max(-1)
        output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
        output_scores.masked_scatter_(output_masks, _scores[output_masks])

        if history is not None:
            history.append(output_tokens.clone())

        # skeptical decoding (depend on the maximum decoding steps.)
        if (step + 1) < max_step:
            skeptical_mask = _skeptical_unmasking(
                output_scores, output_tokens.ne(self.pad), 1 - (step + 1) / max_step
            )

            output_tokens.masked_fill_(skeptical_mask, self.unk)
            output_scores.masked_fill_(skeptical_mask, 0.0)

            if history is not None:
                history.append(output_tokens.clone())
        return output_tokens, output_scores

    def easy_first(self, output_tokens, output_scores, encoder_out, history, step, max_step):
        if step == 0:
            output_masks = output_tokens.eq(self.unk)
            _scores, _tokens = self.decoder(
                normalize=True,
                prev_output_tokens=output_tokens,
                encoder_out=encoder_out,
                masking_type='token_masking',
            ).max(-1)
            output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
            output_scores.masked_scatter_(output_masks, _scores[output_masks])
            _, sorted_ordering = output_scores.sort(1, descending=True)
            _, gen_order = sorted_ordering.sort(1)
            history.append(gen_order)
        else:
            gen_order = history[1]
            output_masks = output_tokens.ne(self.bos) & output_tokens.ne(
                self.eos) & output_tokens.ne(self.pad)
            _scores, _tokens = self.decoder(
                normalize=True,
                prev_output_tokens=output_tokens,
                encoder_out=encoder_out,
                masking_type='easy_first_masking',
                gen_order = gen_order,
            ).max(-1)
            output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
            output_scores.masked_scatter_(output_masks, _scores[output_masks])
        if history is not None:
            history.append(output_tokens.clone())

        return output_tokens, output_scores
            



    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = NATransformerDiscoDecoder(args, tgt_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            decoder.apply(init_bert_params)
        return decoder


@register_model_architecture("disco_transformer", "disco_transformer")
def disco_base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", True)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.apply_bert_init = getattr(args, "apply_bert_init", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    # --- special arguments ---
    args.sg_length_pred = getattr(args, "sg_length_pred", False)
    args.pred_length_offset = getattr(args, "pred_length_offset", False)
    args.length_loss_factor = getattr(args, "length_loss_factor", 0.1)
    args.ngram_predictor = getattr(args, "ngram_predictor", 1)
    args.src_embedding_copy = getattr(args, "src_embedding_copy", False)


@register_model_architecture("disco_transformer", "disco_transformer_wmt_en_de")
def disco_wmt_en_de(args):
    disco_base_architecture(args)
