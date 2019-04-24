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

from fairseq.models.transformer import TransformerDecoder, TransformerModel, TransformerEncoder
from fairseq.models.transformer import Embedding as FairseqEmbedding
from fairseq.models.transformer import transformer_iwslt_de_en
from fairseq.sequence_generator import SequenceGenerator

from unsupervised_translation.fseq_wrappers.fseq_beam_search import FairseqBeamSearchWrapper
from unsupervised_translation.fseq_wrappers.stub import ArgsStub, DictStub

# TODO: get vocab namespaces right
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
                 lang2_namespace: str = "tokens",
                 use_bleu: bool = True) -> None:
        super().__init__(vocab)
        self._lang1_namespace = lang2_namespace  # TODO: DO NOT HARDCODE IT
        self._lang2_namespace = lang2_namespace

        # TODO: do not hardcore this
        self._backtranslation_src_langs = ["en", "ru"]
        self._coeff_denoising = 1
        self._coeff_backtranslation = 1
        self._coeff_translation = 1

        self._label_smoothing = 0.1

        self._pad_index_lang1 = vocab.get_token_index(DEFAULT_PADDING_TOKEN, self._lang1_namespace)
        self._oov_index_lang1 = vocab.get_token_index(DEFAULT_OOV_TOKEN, self._lang1_namespace)
        self._end_index_lang1 = self.vocab.get_token_index(END_SYMBOL, self._lang1_namespace)

        self._pad_index_lang2 = vocab.get_token_index(DEFAULT_PADDING_TOKEN, self._lang2_namespace)
        self._oov_index_lang2 = vocab.get_token_index(DEFAULT_OOV_TOKEN, self._lang2_namespace)
        self._end_index_lang2 = self.vocab.get_token_index(END_SYMBOL, self._lang2_namespace)

        self._reader = dataset_reader
        self._langs_list = self._reader._langs_list
        self._ae_steps = self._reader._ae_steps
        self._bt_steps = self._reader._bt_steps
        self._para_steps = self._reader._para_steps

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
        num_tokens_lang1 = self.vocab.get_vocab_size(self._lang1_namespace)
        num_tokens_lang2 = self.vocab.get_vocab_size(self._lang2_namespace)

        args.share_decoder_input_output_embed = False  # TODO implement shared embeddings

        lang1_dict = DictStub(num_tokens=num_tokens_lang1,
                              pad=self._pad_index_lang1,
                              unk=self._oov_index_lang1,
                              eos=self._end_index_lang1)

        lang2_dict = DictStub(num_tokens=num_tokens_lang2,
                              pad=self._pad_index_lang2,
                              unk=self._oov_index_lang2,
                              eos=self._end_index_lang2)

        # instantiate fairseq classes
        emb_golden_tokens = FairseqEmbedding(num_tokens_lang2, args.decoder_embed_dim, self._pad_index_lang2)

        self._encoder = TransformerEncoder(args, lang1_dict, self._source_embedder)
        self._decoder = TransformerDecoder(args, lang2_dict, emb_golden_tokens)
        self._model = TransformerModel(self._encoder, self._decoder)

        # TODO: do not hardcode max_len_b and beam size
        self._sequence_generator_greedy = FairseqBeamSearchWrapper(SequenceGenerator(tgt_dict=lang2_dict, beam_size=1,
                                                                                     max_len_b=20))
        self._sequence_generator_beam = FairseqBeamSearchWrapper(SequenceGenerator(tgt_dict=lang2_dict, beam_size=7,
                                                                                   max_len_b=20))

    @overrides
    def forward(self,  # type: ignore
                lang_pair: List[str],
                lang1_tokens: Dict[str, torch.LongTensor] = None,
                lang1_golden: Dict[str, torch.LongTensor] = None,
                lang2_tokens: Dict[str, torch.LongTensor] = None,
                lang2_golden: Dict[str, torch.LongTensor] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        """
        # detect training mode and what kind of task we need to compute            
        if lang2_tokens is None and lang1_tokens is None:
            raise ConfigurationError("source_tokens and target_tokens can not both be None")

        mode_training = self.training
        mode_validation = not self.training and lang2_tokens is not None  # change 'target_tokens' condition
        mode_prediction = lang2_tokens is None  # change 'target_tokens' condition

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
                loss = self._forward_seq2seq(lang_pair, lang1_tokens, lang2_tokens, lang2_golden)
                if self._bleu:
                    predicted_indices = self._sequence_generator_beam.generate([self._model],
                                                                               lang1_tokens,
                                                                               self._get_true_pad_mask(lang1_tokens),
                                                                               self._end_index_lang2)
                    predicted_strings = self._indices_to_strings(predicted_indices)
                    golden_strings = self._indices_to_strings(lang2_tokens["tokens"])
                    golden_strings = self._remove_pad_eos(golden_strings)
                    # print(golden_strings, predicted_strings)
                    self._bleu(corpus_bleu(golden_strings, predicted_strings))
            elif task_denoising:  # might need to split it into two blocks for interlingua loss
                loss = self._forward_seq2seq(lang_pair, lang1_tokens, lang2_tokens, lang2_golden)
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
                        predicted_indices = self._sequence_generator_greedy.generate(
                            [self._model], lang2_tokens, self._get_true_pad_mask(lang2_tokens), self._end_index_lang2)
                    model_input = self._strings_to_batch(self._indices_to_strings(predicted_indices), lang2_tokens,
                                                         lang2_golden, curr_lang_pair)
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
                                  self._forward_seq2seq(lang_pair, lang1_tokens, lang2_tokens, lang2_golden)
            if self._bleu:
                predicted_indices = self._sequence_generator_greedy.generate([self._model],
                                                                           lang1_tokens,
                                                                           self._get_true_pad_mask(lang1_tokens),
                                                                           self._end_index_lang2)
                predicted_strings = self._indices_to_strings(predicted_indices)
                golden_strings = self._indices_to_strings(lang2_tokens["tokens"])
                golden_strings = self._remove_pad_eos(golden_strings)
                print(golden_strings, predicted_strings)
                self._bleu(corpus_bleu(golden_strings, predicted_strings))

        elif mode_prediction:
            # TODO: pass target language (in the fseq_encoder append embedded target language to the encoder out)
            predicted_indices = self._sequence_generator_beam.generate([self._model],
                                                                       lang1_tokens,
                                                                       self._get_true_pad_mask(lang1_tokens),
                                                                       self._end_index_lang2)
            output_dict["predicted_indices"] = predicted_indices
            output_dict["predicted_strings"] = self._indices_to_strings(predicted_indices)

        return output_dict

    def _get_true_pad_mask(self, indexed_input):
        mask = util.get_text_field_mask(indexed_input)
        # TODO: account for cases when text field mask doesn't work, like BERT
        return mask

    def _remove_pad_eos(self, golden_strings):
        tmp = []
        for x in golden_strings:
            tmp.append(list(filter(lambda a: a != DEFAULT_PADDING_TOKEN and a != END_SYMBOL, x)))
        return tmp

    def _convert_to_sentences(self, golden_strings, predicted_strings):
        golden_strings_nopad = []
        for s in golden_strings:
            s_nopad = list(filter(lambda t: t != DEFAULT_PADDING_TOKEN, s))
            s_nopad = " ".join(s_nopad)
            golden_strings_nopad.append(s_nopad)
        predicted_strings = [" ".join(s) for s in predicted_strings]
        return golden_strings_nopad, predicted_strings

    def _forward_seq2seq(self, lang_pair: List[str],
                         source_tokens: Dict[str, torch.LongTensor],
                         target_tokens: Dict[str, torch.LongTensor],
                         target_golden: Dict[str, torch.LongTensor]) -> Dict[str, torch.Tensor]:
        source_tokens_padding_mask = self._get_true_pad_mask(source_tokens)
        encoder_out = self._encoder.forward(source_tokens, source_tokens_padding_mask)
        logits, _ = self._decoder.forward(target_tokens["tokens"], encoder_out)
        loss = self._get_ce_loss(logits, target_golden)
        return loss

    def _get_ce_loss(self, logits, golden):
        target_mask = util.get_text_field_mask(golden)
        loss = util.sequence_cross_entropy_with_logits(logits, golden["golden_tokens"], target_mask,
                                                       label_smoothing=self._label_smoothing)
        return loss

    def _indices_to_strings(self, indices: torch.Tensor):
        all_predicted_tokens = []
        for hyp in indices:
            predicted_tokens = [self.vocab.get_token_from_index(idx.item(), namespace=self._lang2_namespace)
                                for idx in hyp]
            all_predicted_tokens.append(predicted_tokens)
        return all_predicted_tokens
        
    def _strings_to_batch(self, source_tokens: List[List[str]], target_tokens: Dict[str, torch.Tensor],
                          target_golden: Dict[str, torch.Tensor], lang_pair: str):
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
        model_input = {"source_tokens": source_batch["tokens"],
                       "target_golden": target_golden,
                       "target_tokens": target_tokens,
                       "lang_pair": lang_pairs}

        return model_input

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics: Dict[str, float] = {}
        if self._bleu and not self.training:
            all_metrics.update({"BLEU": self._bleu.get_metric(reset=reset)})
        return all_metrics