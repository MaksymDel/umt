from typing import Dict, List, Tuple, Any

import numpy
from overrides import overrides
import torch
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
from torch.nn.modules.rnn import LSTMCell

from allennlp.common.checks import ConfigurationError
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.vocabulary import DEFAULT_OOV_TOKEN, DEFAULT_PADDING_TOKEN
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

from fairseq.models.transformer import TransformerDecoder, TransformerModel
from fairseq.models.transformer import Embedding as FairseqEmbedding
from fairseq.models.transformer import base_architecture, transformer_iwslt_de_en
from fairseq.sequence_generator import SequenceGenerator

from unsupervised_translation.modules.fseq_transformer_encoder import AllennlpTransformerEncoder

@Model.register("unsupervised_translation")
class UnsupervisedTranslation(Model):
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
        # TODO: do not hardcore this
        self._backtranslation_src_langs = ["en", "ru"]
        self._coeff_denoising = 1
        self._coeff_backtranslation = 1
        self._coeff_translation = 1

        self._label_smoothing = 0.1

        self._padding_index = vocab.get_token_index(DEFAULT_PADDING_TOKEN, target_namespace) 
        self._oov_index = vocab.get_token_index(DEFAULT_OOV_TOKEN, target_namespace) 
        self._pad_index = vocab.get_token_index(vocab._padding_token, target_namespace)  # pylint: disable=protected-access
        self._start_index = self.vocab.get_token_index(START_SYMBOL, target_namespace)
        self._end_index = self.vocab.get_token_index(END_SYMBOL, target_namespace)

        self._reader = dataset_reader

        self._target_namespace = target_namespace

        # We need the start symbol to provide as the input at the first timestep of decoding, and
        # end symbol as a way to indicate the end of the decoded sequence.
        if use_bleu:
            self._bleu = BLEU(exclude_indices={self._pad_index, self._end_index, self._start_index})
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
        transformer_iwslt_de_en(args)

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

        # stub for useless dictionary that still has to be passed
        class DictStub:
            def __init__(self, num_tokens=None, pad=None, unk=None, eos=None):
                self._num_tokens = num_tokens
                self._pad = pad
                self._unk = unk
                self._eos = eos
            
            def pad(self):
                return self._pad
            
            def unk(self):
                return self._unk

            def eos(self):
                return self._eos

            def __len__(self):
                return self._num_tokens

        src_dict, tgt_dict = DictStub(), DictStub(num_tokens=num_classes, 
                                                  pad=self._padding_index, 
                                                  unk=self._oov_index,
                                                  eos=self._end_index)

        # instantiate fairseq classes
        emb_golden_tokens = FairseqEmbedding(num_classes, args.decoder_embed_dim, self._padding_index)

        self._encoder = AllennlpTransformerEncoder(args, src_dict, self._source_embedder, left_pad=False)
        self._decoder = TransformerDecoder(args, tgt_dict, emb_golden_tokens, left_pad=False)
        self._model = TransformerModel(self._encoder, self._decoder)

        self._sequence_generator_greedy = SequenceGenerator(tgt_dict=tgt_dict, beam_size=1, max_len_b=20)
        self._sequence_generator_beam = SequenceGenerator(tgt_dict=tgt_dict, beam_size=7, max_len_b=20) # TODO: do not hardcode max_len_b and beam size

        ####################################################
        ####################################################
        ####################################################

    @overrides
    def forward(self,  # type: ignore
                lang_pair: List[str],
                source_tokens: Dict[str, torch.LongTensor] = None,
                target_tokens: Dict[str, torch.LongTensor] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        """
        # detect training mode and what kind of task we need to compute            
        if target_tokens is None and source_tokens is None:
            raise ConfigurationError("source_tokens and target_tokens can not both be None")

        mode_training = self.training
        mode_validation = not self.training and target_tokens is not None  # change 'target_tokens' condition
        mode_prediction = target_tokens is None  # change 'target_tokens' condition

        lang_src, lang_tgt = lang_pair[0].split('-')

        if mode_training:
            # task types
            task_translation = False
            task_denoising = False
            task_backtranslation = False

            if lang_src == 'xx':
                task_backtranslation = True
            elif lang_src == lang_tgt:
                task_denoising = True
            elif lang_src != lang_tgt:
                task_translation = True
            else:
                raise ConfigurationError("All tasks are false")

        output_dict = {}
        if mode_training:
            if task_translation or task_denoising:
                # regular cross-entropy loss
                loss = self._forward_seq2seq(lang_pair, source_tokens, target_tokens)
            elif task_backtranslation:
                # our goal is also to learn from regular cross-entropy loss, but since we do not have source tokens,
                # we will generate them ourselves with current model
                    langs_src = self._backtranslation_src_langs.copy()
                    langs_src.remove(lang_tgt)
                    bt_losses = {}
                    for lang_src in langs_src:
                        # TODO: require to pass target language to forward on encoder outputs
                        # We use greedy decoder because it was shown better for backtranslation
                        with torch.no_grad():
                            predictions = \
                                self._sequence_generator_greedy.generate([self._model],
                                                                         self._prepare_fairseq_batch(target_tokens),
                                                                         bos_token=self._start_index)
                            predicted_tokens = self._indices_to_tokens(predictions)
                        # print(predicted_tokens)
                        curr_lang_pair = lang_src + "-" + lang_tgt
                        model_input = self._prepare_batch_input(predicted_tokens, target_tokens, curr_lang_pair)

                        bt_losses['bt:' + curr_lang_pair] = self._forward_seq2seq(**model_input)
            else:
                raise ConfigurationError("No task have been detected")

            if task_translation:
                loss = self._coeff_translation * loss
            elif task_denoising:
                loss = self._coeff_denoising * loss
            elif task_backtranslation:
                loss = 0
                for bt_loss in bt_losses.values():
                    loss += self._coeff_backtranslation * bt_loss

            output_dict["loss"] = loss

        elif mode_validation:
            output_dict["loss"] = self._coeff_translation * \
                                  self._forward_seq2seq(lang_pair, source_tokens, target_tokens)
            if self._bleu:
                predictions = self._sequence_generator_beam.generate([self._model],
                                                                     self._prepare_fairseq_batch(source_tokens),
                                                                     bos_token=self._start_index)
                # shape: (batch_size, beam_size, max_sequence_length)
                # best_predictions = _get_best_predictions(predictions)
                # self._bleu(best_predictions, target_tokens["tokens"])
                output_dict["bleu"] = 0

        elif mode_prediction:
            # TODO: pass target language (in the fseq_encoder append embedded target language to the encoder out)
            predictions = self._sequence_generator_beam.generate([self._model],
                                                                 self._prepare_fairseq_batch(source_tokens),
                                                                 bos_token=self._start_index)
            output_dict["predictions"] = predictions

        return output_dict

    def _forward_seq2seq(self, lang_pair: List[str],
                         source_tokens: Dict[str, torch.LongTensor],
                         target_tokens: Dict[str, torch.LongTensor]) -> Dict[str, torch.Tensor]:
        encoder_out = self._encoder.forward(source_tokens, None)
        logits, _ = self._decoder.forward(target_tokens["tokens"], encoder_out)
        loss = self._get_ce_loss(logits, target_tokens)
        return loss

    def _get_ce_loss(self, logits, target_tokens):
        target_mask = util.get_text_field_mask(target_tokens)
        loss = util.sequence_cross_entropy_with_logits(logits, target_tokens["tokens"], target_mask,
                                                label_smoothing=self._label_smoothing)
        return loss
        
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

    def _prepare_fairseq_batch(self, source_tokens: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        padding_mask = util.get_text_field_mask(source_tokens)
        # source_tokens = source_tokens["tokens"]
        # source_tokens, padding_mask = remove_eos_from_the_beginning(source_tokens, padding_mask)
        lengths = util.get_lengths_from_binary_sequence_mask(padding_mask)
        return {"net_input": {"src_tokens": source_tokens, "src_lengths": lengths}} # TODO: length are ignored even in seq generator; omit it here

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
        output_dict["predicted_tokens"] = self._indices_to_tokens(output_dict["predictions"])
        return output_dict

    def _indices_to_tokens(self, predictions: List[List[Dict[str, torch.LongTensor]]]):
        # TODO: MAKE THIS ACCEPT NORMAL PREDICTIONS AND WRITE FUCN THAT WILL CONVERT FAIRSEQ OUT TO NORMAL ONE
        all_predicted_tokens = []
        for hyps in predictions:
            best_hyp = hyps[0]
            indices = best_hyp["tokens"]

            if self._end_index in indices:
                indices = indices[:((indices == self._end_index).nonzero())]

            predicted_tokens = [self.vocab.get_token_from_index(x.item(), namespace=self._target_namespace)
                                for x in indices]

            all_predicted_tokens.append(predicted_tokens)

        return all_predicted_tokens

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics: Dict[str, float] = {}
        if self._bleu and not self.training:
            all_metrics.update(self._bleu.get_metric(reset=reset))
        return all_metrics