from typing import Dict, List, Tuple, Any

import numpy
from overrides import overrides
import torch
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
from torch.nn.modules.rnn import LSTMCell

from allennlp.common.checks import ConfigurationError
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.attention import LegacyAttention
from allennlp.modules import Attention, TextFieldEmbedder, Seq2SeqEncoder
from allennlp.modules.similarity_functions import SimilarityFunction
from allennlp.models.model import Model
from allennlp.modules.token_embedders import Embedding
from allennlp.nn import util
from allennlp.nn.beam_search import BeamSearch
from allennlp.training.metrics import BLEU
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.dataset import Batch

from fairseq.models.transformer import AllennlpTransformerEncoder, TransformerDecoder, TransformerModel
from fairseq.models.transformer import Embedding as FairseqEmbedding
from fairseq.models.transformer import base_architecture, transformer_iwslt_de_en

@Model.register("unsupervised_translation_fs")
class UnsupervisedTranslationFs(Model):
    """
    This ``SimpleSeq2Seq`` class is a :class:`Model` which takes a sequence, encodes it, and then
    uses the encoded representations to decode another sequence.  You can use this as the basis for
    a neural machine translation system, an abstractive summarization system, or any other common
    seq2seq problem.  The model here is simple, but should be a decent starting place for
    implementing recent models for these tasks.

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        Vocabulary containing source and target vocabularies. They may be under the same namespace
        (`tokens`) or the target tokens can have a different namespace, in which case it needs to
        be specified as `target_namespace`.
    source_embedder : ``TextFieldEmbedder``, required
        Embedder for source side sequences
    encoder : ``Seq2SeqEncoder``, required
        The encoder of the "encoder/decoder" model
    max_decoding_steps : ``int``
        Maximum length of decoded sequences.
    target_namespace : ``str``, optional (default = 'target_tokens')
        If the target side vocabulary is different from the source side's, you need to specify the
        target's namespace here. If not, we'll assume it is "tokens", which is also the default
        choice for the source side, and this might cause them to share vocabularies.
    target_embedding_dim : ``int``, optional (default = source_embedding_dim)
        You can specify an embedding dimensionality for the target side. If not, we'll use the same
        value as the source embedder's.
    attention : ``Attention``, optional (default = None)
        If you want to use attention to get a dynamic summary of the encoder outputs at each step
        of decoding, this is the function used to compute similarity between the decoder hidden
        state and encoder outputs.
    attention_function: ``SimilarityFunction``, optional (default = None)
        This is if you want to use the legacy implementation of attention. This will be deprecated
        since it consumes more memory than the specialized attention modules.
    beam_size : ``int``, optional (default = None)
        Width of the beam for beam search. If not specified, greedy decoding is used.
    scheduled_sampling_ratio : ``float``, optional (default = 0.)
        At each timestep during training, we sample a random number between 0 and 1, and if it is
        not less than this value, we use the ground truth labels for the whole batch. Else, we use
        the predictions from the previous time step for the whole batch. If this value is 0.0
        (default), this corresponds to teacher forcing, and if it is 1.0, it corresponds to not
        using target side ground truth labels.  See the following paper for more information:
        `Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks. Bengio et al.,
        2015 <https://arxiv.org/abs/1506.03099>`_.
    use_bleu : ``bool``, optional (default = True)
        If True, the BLEU metric will be calculated during validation.
    """

    def __init__(self,
                 vocab: Vocabulary,
                 dataset_reader: DatasetReader,
                 source_embedder: TextFieldEmbedder,
                 target_namespace: str = "tokens",
                 use_bleu: bool = True) -> None:
        super().__init__(vocab)

        self._label_smoothing = 0.1
        self._padding_index = 0 # this is always true for allennlp

        self._reader = dataset_reader

        self._target_namespace = target_namespace

        # We need the start symbol to provide as the input at the first timestep of decoding, and
        # end symbol as a way to indicate the end of the decoded sequence.
        self._start_index = self.vocab.get_token_index(START_SYMBOL, self._target_namespace)
        self._end_index = self.vocab.get_token_index(END_SYMBOL, self._target_namespace)

        if use_bleu:
            pad_index = self.vocab.get_token_index(self.vocab._padding_token, self._target_namespace)  # pylint: disable=protected-access
            self._bleu = BLEU(exclude_indices={pad_index, self._end_index, self._start_index})
        else:
            self._bleu = None

        ####################################################
        ####################################################
        # DEFINE FAIRSEQ ENCODER, DECODER, AND MODEL:
        ####################################################
        ####################################################


        # get transformer parameters together
        class ArgsStub:
            def __init__(self):
                pass

        args = ArgsStub()
        args = transformer_iwslt_de_en(args)

        # build encoder
        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = 1024
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = 1024


        # Dense embedding of source vocab tokens.
        self._source_embedder = source_embedder

        # Dense embedding of vocab words in the target space.
        num_classes = self.vocab.get_vocab_size(self._target_namespace)
        args.share_decoder_input_output_embed = False # TODO implement shared embeddings
        self._target_embedder = Embedding(num_classes, args.decoder_embed_dim)

        # stub for useless dictionary that still has to be passed
        class DictStub:
            def  __init__(self, num_classes=None):
                self._num_classes = num_classes

            def __len__(self):
                return self._num_classes

        src_dict, tgt_dict = DictStub(), DictStub(num_classes=num_classes)

        # instantiate fairseq classes

        emb_golden_tokens = FairseqEmbedding(num_classes, target_embedding_dim, self._padding_index)

        self._encoder = AllennlpTransformerEncoder(args, src_dict, self._source_embedder, left_pad=False)
        self._decoder = TransformerDecoder(args, tgt_dict, emb_golden_tokens, left_pad=False)
        self._model = TransformerModel(self._encoder, self._decoder)

        ####################################################
        ####################################################
        ####################################################


    @overrides
    def forward(self,  # type: ignore
                lang_pair: List[str],
                source_tokens: Dict[str, torch.LongTensor],
                target_tokens: Dict[str, torch.LongTensor] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        """
        #print(lang_pair[0])

        lang_src, lang_tgt = lang_pair[0].split('-')

        is_training = self.training
        is_validation = not self.training and target_tokens is not None # change 'target_tokens' condition
        is_prediction = not self.training and target_tokens is None # change 'target_tokens' condition

        output_dict = {}
        if is_training:
            is_para = lang_src != lang_tgt

            if is_para:
                encoder_out = self._encoder.forward(source_tokens, None)
                logits, _ = self._decoder.forward(target_tokens, encoder_out)

                target_mask = util.get_text_field_mask(tokens_B)

                loss = util.sequence_cross_entropy_with_logits(logits, relevant_targets, target_mask,
                                                label_smoothing=self._label_smoothing)
                output_dict = {"loss": total_unsupervised_loss}
            else: # we got to learn from unsupervised objectives (denoising + backtransaltion)
                # 0) learn from denoising
                state = self._encode(source_tokens)
                state = self._init_decoder_state(state)
                output_dict_denoising = self._forward_loop(state, target_tokens)
                loss_denoising = output_dict_denoising["loss"]

                # 1) learn from backtranslation
                # 1.1) generate fake source and prepare model input
                with torch.no_grad():
                    real_target = target_tokens
                    bt_state = self._encode(real_target)
                    bt_state = self._init_decoder_state(bt_state)
                    # TODO: require to pass target language to forward on encoder outputs
                    bt_from = lang_tgt
                    if bt_from == "ru":
                        bt_into = "en"
                    elif bt_from == "en":
                        bt_into = "ru"

                    bt_output_dict = self._forward_loop(bt_state) # translate in inference greedy mode
                    bt_output_dict = self.decode(bt_output_dict)

                    lang_pair = bt_into + '-' + bt_from
                    model_input = self._prepare_batch_input(bt_output_dict['predicted_tokens'], target_tokens, lang_pair)

                # 1.2) learn from newly created sentence pair
                output_dict_bt = self.forward(**model_input)
                loss_bt = output_dict_bt["loss"]

                #stubs
                coeff_denoising = 1
                coeff_bt = 1

                total_unsupervised_loss = coeff_denoising * loss_denoising + coeff_bt * loss_bt
                output_dict = {"loss": total_unsupervised_loss}

        elif is_validation:
            state = self._encode(source_tokens)
            state = self._init_decoder_state(state)
            output_dict = self._forward_beam_search(state) # do not use target tokens
            if target_tokens and self._bleu:
                # shape: (batch_size, beam_size, max_sequence_length)
                top_k_predictions = output_dict["predictions"]
                # shape: (batch_size, max_predicted_sequence_length)
                best_predictions = top_k_predictions[:, 0, :]
                self._bleu(best_predictions, target_tokens["tokens"])

        elif is_prediction:
            state = self._encode(source_tokens)
            state = self._init_decoder_state(state)
            output_dict = self._forward_beam_search(state) # do not use target tokens

        return output_dict


    def _prepare_batch_input(self, source_tokens: List[List[str]], target_tensor_dict: Dict[str, torch.Tensor], lang_pair: str):
        """
        Converts list of sentences which are itself lists of strings into Batch
        suitable for passing into model's forward function.

        TODO: Make sure the right device (CPU/GPU) is used. Predicted tokens might get copied on
        CPU in `self.decode` method...
        """
        # convert source tokens into source tensor_dict
        instances = []
        lang_pairs = []
        for sentence in source_tokens:
            sentence = " ".join(sentence)
            instances.append(self._reader.string_to_instance(sentence))
            lang_pairs.append(lang_pair)

        source_batch = Batch(instances)
        source_batch.index_instances(self.vocab)
        source_batch = source_batch.as_tensor_dict()
        model_input = {"source_tokens": source_batch["source_tokens"],
                       "target_tokens": target_tensor_dict,
                       "lang_pair": lang_pairs}

        return model_input

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Finalize predictions.

        This method overrides ``Model.decode``, which gets called after ``Model.forward``, at test
        time, to finalize predictions. The logic for the decoder part of the encoder-decoder lives
        within the ``forward`` method.

        This method trims the output predictions to the first end symbol, replaces indices with
        corresponding tokens, and adds a field called ``predicted_tokens`` to the ``output_dict``.
        """
        predicted_indices = output_dict["predictions"]
        if not isinstance(predicted_indices, numpy.ndarray):
            predicted_indices = predicted_indices.detach().cpu().numpy()
        all_predicted_tokens = []
        for indices in predicted_indices:
            # Beam search gives us the top k results for each source sentence in the batch
            # but we just want the single best.
            if len(indices.shape) > 1:
                indices = indices[0]
            indices = list(indices)
            # Collect indices till the first end_symbol
            if self._end_index in indices:
                indices = indices[:indices.index(self._end_index)]
            predicted_tokens = [self.vocab.get_token_from_index(x, namespace=self._target_namespace)
                                for x in indices]
            all_predicted_tokens.append(predicted_tokens)
        output_dict["predicted_tokens"] = all_predicted_tokens
        return output_dict


    def _forward_loop(self,
                      state: Dict[str, torch.Tensor],
                      target_tokens: Dict[str, torch.LongTensor] = None) -> Dict[str, torch.Tensor]:
        """
        Make forward pass during training or do greedy search during prediction.

        Notes
        -----
        We really only use the predictions from the method to test that beam search
        with a beam size of 1 gives the same results.
        """
        # shape: (batch_size, max_input_sequence_length)
        source_mask = state["source_mask"]

        batch_size = source_mask.size()[0]

        if target_tokens:
            # shape: (batch_size, max_target_sequence_length)
            targets = target_tokens["tokens"]

            _, target_sequence_length = targets.size()

            # The last input from the target is either padding or the end symbol.
            # Either way, we don't have to process it.
            num_decoding_steps = target_sequence_length - 1
        else:
            num_decoding_steps = self._max_decoding_steps

        # Initialize target predictions with the start index.
        # shape: (batch_size,)
        last_predictions = source_mask.new_full((batch_size,), fill_value=self._start_index)

        step_logits: List[torch.Tensor] = []
        step_predictions: List[torch.Tensor] = []
        for timestep in range(num_decoding_steps):
            if self.training and torch.rand(1).item() < self._scheduled_sampling_ratio:
                # Use gold tokens at test time and at a rate of 1 - _scheduled_sampling_ratio
                # during training.
                # shape: (batch_size,)
                input_choices = last_predictions
            elif not target_tokens:
                # shape: (batch_size,)
                input_choices = last_predictions
            else:
                # shape: (batch_size,)
                input_choices = targets[:, timestep]

            # shape: (batch_size, num_classes)
            output_projections, state = self._prepare_output_projections(input_choices, state)

            # list of tensors, shape: (batch_size, 1, num_classes)
            step_logits.append(output_projections.unsqueeze(1))

            # shape: (batch_size, num_classes)
            class_probabilities = F.softmax(output_projections, dim=-1)

            # shape (predicted_classes): (batch_size,)
            _, predicted_classes = torch.max(class_probabilities, 1)

            # shape (predicted_classes): (batch_size,)
            last_predictions = predicted_classes

            step_predictions.append(last_predictions.unsqueeze(1))

        # shape: (batch_size, num_decoding_steps)
        predictions = torch.cat(step_predictions, 1)

        output_dict = {"predictions": predictions}

        if target_tokens:
            # shape: (batch_size, num_decoding_steps, num_classes)
            logits = torch.cat(step_logits, 1)

            # Compute loss.
            target_mask = util.get_text_field_mask(target_tokens)
            loss = self._get_loss(logits, targets, target_mask)
            output_dict["loss"] = loss

        return output_dict

    def _forward_beam_search(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Make forward pass during prediction using a beam search."""
        batch_size = state["source_mask"].size()[0]
        start_predictions = state["source_mask"].new_full((batch_size,), fill_value=self._start_index)

        # shape (all_top_k_predictions): (batch_size, beam_size, num_decoding_steps)
        # shape (log_probabilities): (batch_size, beam_size)
        all_top_k_predictions, log_probabilities = self._beam_search.search(
                start_predictions, state, self.take_step)

        output_dict = {
                "class_log_probabilities": log_probabilities,
                "predictions": all_top_k_predictions,
        }
        return output_dict

    def _prepare_output_projections(self,
                                    last_predictions: torch.Tensor,
                                    state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:  # pylint: disable=line-too-long
        """
        Decode current state and last prediction to produce produce projections
        into the target space, which can then be used to get probabilities of
        each target token for the next step.

        Inputs are the same as for `take_step()`.
        """
        # shape: (group_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs = state["encoder_outputs"]

        # shape: (group_size, max_input_sequence_length)
        source_mask = state["source_mask"]

        # shape: (group_size, decoder_output_dim)
        decoder_hidden = state["decoder_hidden"]

        # shape: (group_size, decoder_output_dim)
        decoder_context = state["decoder_context"]

        # shape: (group_size, target_embedding_dim)
        embedded_input = self._target_embedder(last_predictions)

        if self._attention:
            # shape: (group_size, encoder_output_dim)
            attended_input = self._prepare_attended_input(decoder_hidden, encoder_outputs, source_mask)

            # shape: (group_size, decoder_output_dim + target_embedding_dim)
            decoder_input = torch.cat((attended_input, embedded_input), -1)
        else:
            # shape: (group_size, target_embedding_dim)
            decoder_input = embedded_input

        # shape (decoder_hidden): (batch_size, decoder_output_dim)
        # shape (decoder_context): (batch_size, decoder_output_dim)
        decoder_hidden, decoder_context = self._decoder_cell(
                decoder_input,
                (decoder_hidden, decoder_context))

        state["decoder_hidden"] = decoder_hidden
        state["decoder_context"] = decoder_context

        # shape: (group_size, num_classes)
        output_projections = self._output_projection_layer(decoder_hidden)

        return output_projections, state

    def _prepare_attended_input(self,
                                decoder_hidden_state: torch.LongTensor = None,
                                encoder_outputs: torch.LongTensor = None,
                                encoder_outputs_mask: torch.LongTensor = None) -> torch.Tensor:
        """Apply attention over encoder outputs and decoder state."""
        # Ensure mask is also a FloatTensor. Or else the multiplication within
        # attention will complain.
        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs_mask = encoder_outputs_mask.float()

        # shape: (batch_size, max_input_sequence_length)
        input_weights = self._attention(
                decoder_hidden_state, encoder_outputs, encoder_outputs_mask)

        # shape: (batch_size, encoder_output_dim)
        attended_input = util.weighted_sum(encoder_outputs, input_weights)

        return attended_input

    @staticmethod
    def _get_loss(logits: torch.LongTensor,
                  targets: torch.LongTensor,
                  target_mask: torch.LongTensor) -> torch.Tensor:
        """
        Compute loss.

        Takes logits (unnormalized outputs from the decoder) of size (batch_size,
        num_decoding_steps, num_classes), target indices of size (batch_size, num_decoding_steps+1)
        and corresponding masks of size (batch_size, num_decoding_steps+1) steps and computes cross
        entropy loss while taking the mask into account.

        The length of ``targets`` is expected to be greater than that of ``logits`` because the
        decoder does not need to compute the output corresponding to the last timestep of
        ``targets``. This method aligns the inputs appropriately to compute the loss.

        During training, we want the logit corresponding to timestep i to be similar to the target
        token from timestep i + 1. That is, the targets should be shifted by one timestep for
        appropriate comparison.  Consider a single example where the target has 3 words, and
        padding is to 7 tokens.
           The complete sequence would correspond to <S> w1  w2  w3  <E> <P> <P>
           and the mask would be                     1   1   1   1   1   0   0
           and let the logits be                     l1  l2  l3  l4  l5  l6
        We actually need to compare:
           the sequence           w1  w2  w3  <E> <P> <P>
           with masks             1   1   1   1   0   0
           against                l1  l2  l3  l4  l5  l6
           (where the input was)  <S> w1  w2  w3  <E> <P>
        """
        # shape: (batch_size, num_decoding_steps)
        relevant_targets = targets[:, 1:].contiguous()

        # shape: (batch_size, num_decoding_steps)
        relevant_mask = target_mask[:, 1:].contiguous()

        return util.sequence_cross_entropy_with_logits(logits, relevant_targets, relevant_mask)

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics: Dict[str, float] = {}
        if self._bleu and not self.training:
            all_metrics.update(self._bleu.get_metric(reset=reset))
        return all_metrics