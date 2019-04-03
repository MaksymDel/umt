from typing import Dict, List, Tuple

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
from allennlp.modules import Attention, TextFieldEmbedder, Seq2SeqEncoder, TokenEmbedder
from allennlp.modules.similarity_functions import SimilarityFunction
from allennlp.models.model import Model
from allennlp.modules.token_embedders import Embedding
from allennlp.nn import util
from allennlp.nn.beam_search import BeamSearch
from allennlp.training.metrics import BLEU
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.dataset import Batch

# @Model.register("unsupervised_translation")
class UnsupervisedTranslation(Model):
    """

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        Vocabulary containing source and target vocabularies. They may be under the same namespace
        (`tokens`) or the target tokens can have a different namespace, in which case it needs to
        be specified as `target_namespace`.

    """

    def __init__(self,
                 vocab: Vocabulary,
                 dataset_reader: DatasetReader,
                 lang_tag_embedder: TokenEmbedder,
                 use_bleu: bool = True) -> None:
        super().__init__(vocab)

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

    # ABSTRACT
    def _run_encoder(self, source_tokens: Dict[str, torch.LongTensor]):
        raise NotImplementedError("Please Implement this method in your unsupervised translation model subclass")

    # ABSTRACT
    def _run_decoder(self,
                    encoded_input: torch.FloatTensor,
                    target_tokens: Dict[str, torch.LongTensor]):
        raise NotImplementedError("Please Implement this method in your unsupervised translation model subclass")

    # ABSTRACT
    def _run_decoder_beam(self,
                         encoded_input: torch.FloatTensor):
        raise NotImplementedError("Please Implement this method in your unsupervised translation model subclass")


    @overrides
    def forward(self,  # type: ignore
                lang_pair: List[str],
                source_tag: torch.LongTensor,
                target_tag: torch.LongTensor,
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

        if is_training:
            is_para = lang_src != lang_tgt

            if is_para:
                state = self._run_encoder(source_tokens)
                state = self._attach_tgt_lang_tag(state, target_tag)
                output_dict = self._run_decoder(state, target_tokens)
            else: # we got to learn from unsupervised objectives (denoising + backtransaltion)
                # 0) learn from denoising
                state = self._run_encoder(source_tokens)
                state = self._attach_tgt_lang_tag(state, target_tag)
                output_dict_denoising = self._run_decoder(state, target_tokens)
                loss_denoising = output_dict_denoising["loss"]

                # 1) learn from backtranslation
                # 1.1) generate fake source and prepare model input
                with torch.no_grad():
                    # TODO: require to pass target language to forward on encoder outputs
                    bt_from = lang_tgt
                    if bt_from == "ru":
                        bt_into = "en"
                    elif bt_from == "en":
                        bt_into = "ru"

                    batch_size = state.size()[0]

                    bt_into_indexed = self._index_lang_tag(bt_into)

                    real_target = target_tokens
                    bt_state = self._run_encoder(real_target)
                    bt_state = self._attach_tgt_lang_tag(bt_state, bt_into_indexed)
                    bt_output_dict = self._run_decoder(bt_state) # translate in inference greedy mode
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
            state = self._run_encoder(source_tokens)
            state = selfself._attach_tgt_lang_tag(state, target_tag)
            output_dict = self._forward_beam_search(state) # do not use target tokens
            if target_tokens and self._bleu:
                # shape: (batch_size, beam_size, max_sequence_length)
                top_k_predictions = output_dict["predictions"]
                # shape: (batch_size, max_predicted_sequence_length)
                best_predictions = top_k_predictions[:, 0, :]
                self._bleu(best_predictions, target_tokens["tokens"])

        elif is_prediction:
            state = self._run_encoder(source_tokens)
            state = selfself._attach_tgt_lang_tag(state, target_tag)
            output_dict = self._forward_beam_search(state) # do not use target tokens

        return output_dict


    def _prepare_batch_tag_idx(tag: str):
        tag_idx
        X = torch.randn(100, 700)
X = X.unsqueeze(2).repeat(1, 1, 28)

    def _index_lang_tag(self, tag: str):
        pass


    def _attach_tgt_lang_tag(self,
                            encoded_source: torch.FloatTensor,
                            indexed_lang_tag: torch.LongTensor):
        embedded_lang_tag = lang_tag_embedder(indexed_lang_tag)
        return encoded_source # TODO: impelement concatenation


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

        src_tag, tgt_tag = lang_pair.split("-")
        src_tag_idx, tgt_tag_idx = self._index_lang_tag(src_tag), self._index_lang_tag(tgt_tag)

        src_tags_idx = []
        tgt_tags_idx = []
        for sentence in source_tokens:
            sentence = " ".join(sentence)
            instances.append(self._reader.string_to_instance(sentence))
            lang_pairs.append(lang_pair)
            src_tags_idx.append(src_tag_idx)
            tgt_tags_idx.append(tgt_tag_idx)

        source_batch = Batch(instances)
        source_batch.index_instances(self.vocab)
        source_batch = source_batch.as_tensor_dict()


        model_input = {"source_tokens": source_batch["source_tokens"],
                       "target_tokens": target_tensor_dict,
                       "lang_pair": lang_pairs,
                       "source_tag": src_tags_idx,
                       "target_tag": tgt_tags_idx}

        return model_input

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics: Dict[str, float] = {}
        if self._bleu and not self.training:
            all_metrics.update(self._bleu.get_metric(reset=reset))
        return all_metrics