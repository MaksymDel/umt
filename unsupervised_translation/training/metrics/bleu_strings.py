from collections import Counter
import math
from typing import Iterable, Tuple, Dict, Set, List

from overrides import overrides
import torch
from nltk.translate.bleu_score import corpus_bleu

from allennlp.training.metrics.metric import Metric


@Metric.register("bleu_strings")
class BleuStrings(Metric):
    """
    Bilingual Evaluation Understudy (BLEU).

    BLEU is a common metric used for evaluating the quality of machine translations
    against a set of reference translations. See `Papineni et. al.,
    "BLEU: a method for automatic evaluation of machine translation", 2002
    <https://www.semanticscholar.org/paper/8ff93cfd37dced279134c9d642337a2085b31f59/>`_.

    Parameters
    ----------
    ngram_weights : ``Iterable[float]``, optional (default = (0.25, 0.25, 0.25, 0.25))
        Weights to assign to scores for each ngram size.
    exclude_indices : ``Set[int]``, optional (default = None)
        Indices to exclude when calculating ngrams. This should usually include
        the indices of the start, end, and pad tokens.

    Notes
    -----
    We chose to implement this from scratch instead of wrapping an existing implementation
    (such as `nltk.translate.bleu_score`) for a two reasons. First, so that we could
    pass tensors directly to this metric instead of first converting the tensors to lists of strings.
    And second, because functions like `nltk.translate.bleu_score.corpus_bleu()` are
    meant to be called once over the entire corpus, whereas it is more efficient
    in our use case to update the running precision counts every batch.

    This implementation only considers a reference set of size 1, i.e. a single
    gold target sequence for each predicted sequence.
    """
    def __init__(self) -> None:
        self._bleu_score = 0

    @overrides
    def __call__(self,  # type: ignore
                 predictions: List[str],
                 gold_targets: List[str]) -> None:
        """
        Update precision counts.

        Parameters
        ----------
        predictions : ``torch.LongTensor``, required
            Batched predicted tokens of shape `(batch_size, max_sequence_length)`.
        references : ``torch.LongTensor``, required
            Batched reference (gold) translations with shape `(batch_size, max_gold_sequence_length)`.

        Returns
        -------
        None
        """
        self._bleu_score = corpus_bleu(gold_targets, predictions)

    @overrides
    def get_metric(self, reset: bool = False) -> Dict[str, float]:
        bleu = self._bleu_score
        if reset:
            self.reset()
        return {"BLEU": bleu}
