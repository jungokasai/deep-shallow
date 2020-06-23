# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This file implements:
Fast Structured Decoding of Non-autoregressive Transformers
"""
import torch
import torch.nn.functional as F

from fairseq import utils
from fairseq.models import (
register_model,
register_model_architecture,
)
from fairseq.models.nat import (
NATransformerDecoder,
CMLMNATransformerModel,
ensemble_decoder,
)
from fairseq.models.transformer import TransformerDecoder
from fairseq.models.nat import NATransformerDecoder
from fairseq.utils import new_arange
from fairseq.iterative_refinement_generator import DecoderOut
from fairseq.models.fairseq_encoder import EncoderOut
from typing import Any, Dict, List, Optional, Tuple
from torch import Tensor
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from copy import deepcopy


@register_model("cmlm_at")
class CMLMATModel(CMLMNATransformerModel):
    @staticmethod
    def add_args(parser):
        CMLMNATransformerModel.add_args(parser)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = CMLMATDecoder(args, tgt_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            decoder.apply(init_bert_params)
        return decoder

    def forward(
        self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, **kwargs
    ):
        # Because of torch jit, this is never called in inference. 
        # We call encoder.forward and decoder.forward separately
        assert not self.decoder.bottom_nat.src_embedding_copy, "do not support embedding copy."

        # encoding
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)

        # get rid of bos for AT consistency
        prev_output_tokens = prev_output_tokens[:, 1:]
        tgt_tokens = tgt_tokens[:, 1:]


        padded_tgt_lengths = (src_lengths*1.2+10).long() # [B]
        max_tgt_length = padded_tgt_lengths.max()
        bsz, seq_len = tgt_tokens.size()
        prev_output_tokens_new = prev_output_tokens.new_ones([bsz, max_tgt_length])*self.pad
        tgt_tokens_new = tgt_tokens.new_ones([bsz, max_tgt_length])*self.pad

        arange_mat = torch.arange(max_tgt_length, 
            device=prev_output_tokens.device).squeeze(0).repeat([bsz, 1])
        # Fill in eos's
        prev_output_tokens_new = prev_output_tokens_new.masked_fill(
            arange_mat<padded_tgt_lengths.unsqueeze(1), self.eos)
        tgt_tokens_new = tgt_tokens_new.masked_fill(
            arange_mat<padded_tgt_lengths.unsqueeze(1), self.eos)
        # Fill in actual tokens
        non_pad_mask = prev_output_tokens.ne(self.pad)
        prev_output_tokens_new = prev_output_tokens_new.masked_scatter(
            arange_mat<non_pad_mask.sum(1, keepdim=True), prev_output_tokens[non_pad_mask])
        tgt_tokens_new = tgt_tokens_new.masked_scatter(
            arange_mat<non_pad_mask.sum(1, keepdim=True), tgt_tokens[non_pad_mask])


        if self.training:
            # mask eos from left
            nb_eos = prev_output_tokens_new.eq(self.eos).sum(1, keepdim=True)
            nb_masked_eos = (nb_eos.float().uniform_()*nb_eos).long() + 1
            eos_start_idxes = non_pad_mask.sum(1, keepdim=True) - 1
            # -1 for the original eos
            eos_masking = (eos_start_idxes <= arange_mat) & (arange_mat< eos_start_idxes + nb_masked_eos)
            prev_output_tokens_new = prev_output_tokens_new.masked_fill(
                eos_masking, self.unk) 

        prev_output_tokens = prev_output_tokens_new
        tgt_tokens = tgt_tokens_new


        # decoding
        word_ins_out, word_ins_out_at, prev_output_tokens_at = self.decoder(
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out,
            tgt_tokens=tgt_tokens,
            normalize=False,
            )
        word_ins_mask = prev_output_tokens.eq(self.unk)
        word_ins_mask_at = prev_output_tokens_at.ne(self.pad) &  word_ins_mask

        return {
            #"word_ins": {
            #    "out": word_ins_out, "tgt": tgt_tokens,
            #    "mask": word_ins_mask, "ls": self.args.label_smoothing,
            #    "nll_loss": True,
            #    #"factor": 0.5,
            #},
            "word_ins_at": {
                "out": word_ins_out_at, "tgt": tgt_tokens,
                "mask": word_ins_mask_at, "ls": self.args.label_smoothing,
                "nll_loss": True,
                #"factor": 0.5,
            },
        }

#    @torch.jit.export
#    def get_normalized_probs(
#        self,
#        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
#        log_probs: bool,
#        sample: Optional[Dict[str, Tensor]] = None,
#    ):
#        """Get normalized probabilities (or log probs) from a net's output."""
#        return self.get_normalized_probs_scriptable(net_output, log_probs, sample)


class CMLMATDecoder(TransformerDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        # Lite
        lite_args = deepcopy(args)
        lite_args.decoder_layers = 1
        super().__init__(
            lite_args, dictionary, embed_tokens, no_encoder_attn,
            )
        self.bottom_nat = NATransformerDecoder(
            args, dictionary, embed_tokens, no_encoder_attn,
            )
        self.bos = dictionary.bos()
        self.unk = dictionary.unk()
        self.eos = dictionary.eos()
        self.pad = dictionary.pad()

    def forward_at(
        self,
        prev_output_tokens,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
        bottom_features: Optional[Tensor] = None,
        **unused,
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            bottom_features = bottom_features,
        )
        if not features_only:
            x = self.output_layer(x)
        return x, extra

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        bottom_features = None,
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        assert bottom_features is not None
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        # embed positions
        positions = (
            self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )
            if self.embed_positions is not None
            else None
        )
        if incremental_state is not None:
            step = prev_output_tokens.size(1) - 1
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]
                # add bottom features
                if bottom_features is not None:
                    nat_len = bottom_features.size(1)
                    # if we already exceeded predicted length from NAT, skip for now.
                    # TODO: maybe epsilon training?
                    if nat_len > step:
                        bottom_features = bottom_features[:, step:step+1]
                    else:
                        bottom_features = None

        if (bottom_features is not None) and (positions is not None):
            bottom_features = bottom_features*0.0
            positions = positions + bottom_features

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)


        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            encoder_state: Optional[Tensor] = None
            if encoder_out is not None:
                if self.layer_wise_attention:
                    encoder_states = encoder_out.encoder_states
                    assert encoder_states is not None
                    encoder_state = encoder_states[idx]
                else:
                    encoder_state = encoder_out.encoder_out

            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = torch.empty(1).uniform_()
            if not self.training or (dropout_probability > self.decoder_layerdrop):
                x, layer_attn, _ = layer(
                    x,
                    encoder_state,
                    encoder_out.encoder_padding_mask
                    if encoder_out is not None
                    else None,
                    incremental_state,
                    self_attn_mask=self_attn_mask,
                    self_attn_padding_mask=self_attn_padding_mask,
                    need_attn=bool((idx == alignment_layer)),
                    need_head_weights=bool((idx == alignment_layer)),
                )
                inner_states.append(x)
                if layer_attn is not None and idx == alignment_layer:
                    attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": [attn], "inner_states": inner_states}

                    


    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[EncoderOut] = None,
        tgt_tokens: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
        inference: bool = False,
        normalize: bool = True,
        **unused
        ):

        step = prev_output_tokens.size(1) - 1
        if not inference:
            # Training so always extract features
            bottom_features, _ = self.bottom_nat.extract_features(
                prev_output_tokens,
                encoder_out=encoder_out,
                embedding_copy=(step == 0) & self.bottom_nat.src_embedding_copy,
            )
            decoder_out = self.bottom_nat.output_layer(bottom_features)
            prev_output_tokens = tgt_tokens.clone()
            prev_output_tokens[:, 1:] = tgt_tokens[:, :-1]
            prev_output_tokens.masked_fill_(
                prev_output_tokens.eq(self.eos), self.pad
            )
            prev_output_tokens[:, 0] = self.eos
            decoder_out_at, _ = self.forward_at(
                prev_output_tokens,
                encoder_out=encoder_out,
                bottom_features=bottom_features,
                )
            if normalize:
                return F.log_softmax(decoder_out, -1), F.log_softmax(decoder_out_at, -1)
            else:
                return decoder_out, decoder_out_at, prev_output_tokens

        if step == 0 and inference:
            assert not self.training
            # only when this is the first step, run the bottom NAT
            decoder_out = self.initialize_output_tokens(encoder_out, prev_output_tokens)
            # bottom_features: [B, T_tgt, C]
            # encoder_out.encoder_out: [T_src, B, C]
            bottom_features, _ = self.bottom_nat.extract_features(
                decoder_out.output_tokens,
                encoder_out=encoder_out,
                embedding_copy=(step == 0) & self.bottom_nat.src_embedding_copy,
            )
            encoder_out = encoder_out._replace(
                bottom_features = bottom_features,
                )

        else:
            assert not self.training
            # Done with step 0
            #print(step, prev_output_tokens)
            bottom_features = encoder_out.bottom_features

        decoder_out_at = self.forward_at(
                                prev_output_tokens,
                                encoder_out=encoder_out,
                                tgt_tokens=tgt_tokens,
                                incremental_state=incremental_state,
                                features_only=features_only,
                                alignment_layer=alignment_layer,
                                alignment_heads=alignment_heads,
                                src_lengths=src_lengths,
                                return_all_hiddens=return_all_hiddens,
                                bottom_features=bottom_features,
                                )

        return decoder_out_at, encoder_out

    def initialize_output_tokens(self, encoder_out, prev_output_tokens):
        # length prediction
        src_lengths = (~encoder_out.encoder_padding_mask).sum(1, keepdim=True)
        length_tgt = (src_lengths*1.2+10).long()
        max_length_tgt = length_tgt.max()
        bsz =  prev_output_tokens.size(0)

        initial_output_tokens = prev_output_tokens.new_ones([bsz, max_length_tgt])*self.pad
        arange_mat = torch.arange(max_length_tgt, 
            device=prev_output_tokens.device).squeeze(0).repeat([bsz, 1])
        initial_output_tokens = initial_output_tokens.masked_fill(
            arange_mat < length_tgt-1, self.unk)
        initial_output_tokens = initial_output_tokens.masked_fill(
            arange_mat.eq(length_tgt-1), self.eos)

        initial_output_scores = initial_output_tokens.new_zeros(
            *initial_output_tokens.size()
        ).type_as(encoder_out.encoder_out)

        return DecoderOut(
            output_tokens=initial_output_tokens,
            output_scores=initial_output_scores,
            attn=None,
            step=0,
            max_step=10000,
            history=None
        )


@register_model_architecture("cmlm_at", "cmlm_at")
def cmlmat_base_architecture(args):
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


@register_model_architecture("cmlm_at", "cmlm_at_wmt_en_de")
def cmlmat_wmt_en_de(args):
    cmlm_base_architecture(args)
