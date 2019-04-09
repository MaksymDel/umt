from typing import Dict, List

from overrides import overrides
import torch

from allennlp.common.checks import ConfigurationError
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.vocabulary import DEFAULT_OOV_TOKEN, DEFAULT_PADDING_TOKEN
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules import TextFieldEmbedder
from allennlp.models.model import Model
from allennlp.nn import util
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.dataset import Batch
from allennlp.training.metrics.average import Average

from nltk.translate.bleu_score import corpus_bleu

from fairseq.models.transformer import TransformerDecoder, TransformerModel
from fairseq.models.transformer import Embedding as FairseqEmbedding
from fairseq.models.transformer import transformer_iwslt_de_en
from fairseq.sequence_generator import SequenceGenerator

from unsupervised_translation.modules.fseq_transformer_encoder import AllennlpTransformerEncoder
from unsupervised_translation.fseq_wrappers.fseq_beam_search import FairseqBeamSearch
from unsupervised_translation.fseq_wrappers.stub import ArgsStub, DictStub

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

        self._pad_index = vocab.get_token_index(DEFAULT_PADDING_TOKEN, target_namespace)
        self._oov_index = vocab.get_token_index(DEFAULT_OOV_TOKEN, target_namespace)
        self._start_index = self.vocab.get_token_index(START_SYMBOL, target_namespace)
        self._end_index = self.vocab.get_token_index(END_SYMBOL, target_namespace)

        self._reader = dataset_reader

        self._target_namespace = target_namespace

        if use_bleu:
            self._bleu = Average()
        else:
            self._bleu = None

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
        args.share_decoder_input_output_embed = False  # TODO implement shared embeddings

        src_dict, tgt_dict = DictStub(), DictStub(num_tokens=num_classes,
                                                  pad=self._pad_index,
                                                  unk=self._oov_index,
                                                  eos=self._end_index)

        # instantiate fairseq classes
        emb_golden_tokens = FairseqEmbedding(num_classes, args.decoder_embed_dim, self._pad_index)

        self._encoder = AllennlpTransformerEncoder(args, src_dict, self._source_embedder, left_pad=False)
        self._decoder = TransformerDecoder(args, tgt_dict, emb_golden_tokens, left_pad=False)
        self._model = TransformerModel(self._encoder, self._decoder)

        # TODO: do not hardcode max_len_b and beam size
        self._sequence_generator_greedy = FairseqBeamSearch(SequenceGenerator(tgt_dict=tgt_dict, beam_size=1,
                                                                              max_len_b=20))
        self._sequence_generator_beam = FairseqBeamSearch(SequenceGenerator(tgt_dict=tgt_dict, beam_size=7,
                                                                            max_len_b=20))

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

            if task_translation:
                loss = self._forward_seq2seq(lang_pair, source_tokens, target_tokens)
            elif task_denoising:  # might need to split it into two blocks for interlingua loss
                loss = self._forward_seq2seq(lang_pair, source_tokens, target_tokens)
            elif task_backtranslation:
                # our goal is also to learn from regular cross-entropy loss, but since we do not have source tokens,
                # we will generate them ourselves with current model
                langs_src = self._backtranslation_src_langs.copy()
                langs_src.remove(lang_tgt)
                bt_losses = {}
                for lang_src in langs_src:
                    curr_lang_pair = lang_src + "-" + lang_tgt
                    # TODO: require to pass target language to forward on encoder outputs
                    # We use greedy decoder because it was shown better for backtranslation
                    with torch.no_grad():
                        predicted_indices = self._sequence_generator_greedy.generate([self._model], target_tokens,
                                                                                     self._end_index, self._start_index)
                    model_input = self._strings_to_batch(self._indices_to_strings(predicted_indices), target_tokens,
                                                         curr_lang_pair)
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
                predicted_indices = self._sequence_generator_beam.generate([self._model], source_tokens,
                                                                           self._end_index, self._start_index)
                predicted_strings = self._indices_to_strings(predicted_indices)
                golden_strings = self._indices_to_strings(target_tokens["tokens"])
                golden_strings = list(filter(lambda t: t != DEFAULT_PADDING_TOKEN, golden_strings))
                # shape: (batch_size, beam_size, max_sequence_length)
                self._bleu(corpus_bleu(golden_strings, predicted_strings))

        elif mode_prediction:
            # TODO: pass target language (in the fseq_encoder append embedded target language to the encoder out)
            predicted_indices = self._sequence_generator_beam.generate([self._model], source_tokens,
                                                                       self._end_index, self._start_index)
            output_dict["predicted_indices"] = predicted_indices
            output_dict["predicted_strings"] = self._indices_to_strings(predicted_indices)

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

    def _indices_to_strings(self, indices: torch.Tensor):
        all_predicted_tokens = []
        for hyp in indices:
            predicted_tokens = [self.vocab.get_token_from_index(idx.item(), namespace=self._target_namespace)
                                for idx in hyp]
            all_predicted_tokens.append(predicted_tokens)
        return all_predicted_tokens
        
    def _strings_to_batch(self, source_tokens: List[List[str]], target_tensor_dict: Dict[str, torch.Tensor],
                          lang_pair: str):
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
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics: Dict[str, float] = {}
        if self._bleu and not self.training:
            all_metrics.update({"BLEU": self._bleu.get_metric(reset=reset)})
        return all_metrics