from typing import List, Dict
import torch
from fairseq.sequence_generator import SequenceGenerator
from fairseq.models import FairseqModel
from allennlp.nn import util


class FairseqBeamSearch:
    def __init__(self, sequence_generator: SequenceGenerator):
        self._sequence_generator = sequence_generator

    def generate(self, models: List[FairseqModel], indexed_tokens: Dict[str, torch.Tensor],
                 padding_mask: torch.LongTensor, eos_index: int, bos_index: int = None):
        model_input = self._prepare_fairseq_batch(indexed_tokens, padding_mask)
        predictions = self._sequence_generator.generate(models, model_input, bos_token=bos_index)
        indices = self._parse_seqgen_out(predictions, eos_index)
        return indices

    def _prepare_fairseq_batch(self, tokens: Dict[str, torch.Tensor], padding_mask: torch.LongTensor) -> Dict[str, Dict[str, torch.Tensor]]:
        padding_mask = util.get_text_field_mask(tokens)
        # source_tokens = source_tokens["tokens"]
        # source_tokens, padding_mask = remove_eos_from_the_beginning(source_tokens, padding_mask)
        return {"net_input": {"src_tokens": tokens,
                              "src_tokens_padding_mask": padding_mask}}

    def _parse_seqgen_out(self, x, eos_index) -> List[torch.Tensor]:
        all_indices = []
        for hyps in x:
            best_hyp = hyps[0]
            indices = best_hyp["tokens"]
            all_indices.append(indices[:-1])  # remove eos from output
        return all_indices
