from typing import List, Dict
import torch
from fairseq.sequence_generator import SequenceGenerator
from fairseq.models import FairseqModel
from allennlp.nn import util


class FairseqBeamSearch:
    def __init__(self, sequence_generator: SequenceGenerator):
        self._sequence_generator = sequence_generator

    def generate(self, models: List[FairseqModel], indexed_tokens: Dict[str, torch.Tensor],
                 eos_index, bos_index: int = None):
        model_input = self._prepare_fairseq_batch(indexed_tokens)
        predictions = self._sequence_generator.generate(models, model_input, bos_token=bos_index)
        indices = self._parse_seqgen_out(predictions, eos_index)
        return indices

    def _prepare_fairseq_batch(self, tokens: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
        padding_mask = util.get_text_field_mask(tokens)
        # source_tokens = source_tokens["tokens"]
        # source_tokens, padding_mask = remove_eos_from_the_beginning(source_tokens, padding_mask)
        lengths = util.get_lengths_from_binary_sequence_mask(padding_mask)
        return {"net_input": {"src_tokens": tokens,
                              "src_lengths": lengths}}  # TODO: length are ignored even in seq generator; omit it here

    def _parse_seqgen_out(self, x, eos_index) -> List[torch.Tensor]:
        all_indices = []
        for hyps in x:
            best_hyp = hyps[0]
            indices = best_hyp["tokens"]
            all_indices.append(indices)
        return all_indices
