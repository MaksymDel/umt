import math

from allennlp.modules.seq2seq_encoders import StackedSelfAttentionEncoder
from allennlp.modules import TextFieldEmbedder
from allennlp.nn import util

from fairseq.models.fairseq_encoder import FairseqEncoder

class AllennlpTransformerEncoder(FairseqEncoder):
    """
    Fairseq Encoder that uses allennlp's stacked self-attention and allennlp's indexed input

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
        left_pad (bool, optional): whether the input is left-padded
            (default: True).
    """

    def __init__(self, args, dictionary, source_embedder: TextFieldEmbedder, left_pad=False):
        super().__init__(dictionary)

        self._seq2seq_encoder = StackedSelfAttentionEncoder(input_dim= int(source_embedder.get_output_dim()),
                                                    hidden_dim= int(args.encoder_embed_dim),
                                                    projection_dim= int(args.encoder_embed_dim / args.encoder_attention_heads),
                                                    feedforward_hidden_dim= int(args.encoder_ffn_embed_dim),
                                                    num_layers= int(args.encoder_layers),
                                                    num_attention_heads= int(args.encoder_attention_heads),
                                                    use_positional_encoding = True,
                                                    dropout_prob = int(args.dropout),
                                                    residual_dropout_prob = int(args.relu_dropout),
                                                    attention_dropout_prob = int(args.attention_dropout))

        self._source_embedder = source_embedder
        embed_dim = source_embedder.get_output_dim()
        self.embed_scale = math.sqrt(embed_dim)
        self._max_source_positions = args.max_source_positions


    def forward(self, src_tokens, src_lengths):
        """
        Args:
            src_tokens (LongTensor): allennlp's indexed text field
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`; not used actually, so we just pass a stub

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
        """
        # shape: (batch_size, max_input_sequence_length, encoder_input_dim)
        embedded_input = self.embed_scale * self._source_embedder(src_tokens)
        # shape: (batch_size, max_input_sequence_length)
        source_mask = util.get_text_field_mask(src_tokens)
        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs = self._seq2seq_encoder(embedded_input, source_mask)

        # return {
        #     'encoder_out': x,  # T x B x C
        #     'encoder_padding_mask': encoder_padding_mask,  # B x T
        # }

        return {
                "encoder_padding_mask": source_mask == 0, # fairseq uses inverted mask
                "encoder_out": encoder_outputs.transpose(0, 1)
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if encoder_out['encoder_out'] is not None:
            encoder_out['encoder_out'] = \
                encoder_out['encoder_out'].index_select(1, new_order)
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = \
                encoder_out['encoder_padding_mask'].index_select(0, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return self._max_source_positions