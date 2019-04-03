from allennlp.data import DataIterator
from typing import Any, Dict, Iterable, Iterator, List, Set
from allennlp.data.instance import Instance
from allennlp.data.dataset import Batch
from collections import defaultdict

@DataIterator.register("homogeneous_batches")
class HomogeneousBatchIterator(DataIterator):
    """
    An iterator that takes instances of various types
    and yields single-type batches of them. There's a flag
    to allow mixed-type batches, but at that point you might
    as well just use ``BasicIterator``?
    """
    def __init__(self,
                 type_field_name: str = "dataset",
                 allow_mixed_batches: bool = False,
                 batch_size: int = 32) -> None:
        super().__init__(batch_size)
        self.type_field_name = type_field_name
        self.allow_mixed_batches = allow_mixed_batches

    def _create_batches(self, instances: Iterable[Instance], shuffle: bool) -> Iterable[Batch]:
        """
        This method should return one epoch worth of batches.
        """
        hoppers: Dict[Any, List[Instance]] = defaultdict(list)

        for instance in instances:
            # Which hopper do we put this instance in?
            if self.allow_mixed_batches:
                instance_type = ""
            else:
                instance_type = instance.fields[self.type_field_name].metadata  # type: ignore

            hoppers[instance_type].append(instance)

            # If the hopper is full, yield up the batch and clear it.
            if len(hoppers[instance_type]) >= self._batch_size:
                yield Batch(hoppers[instance_type])
                hoppers[instance_type].clear()

        # Deal with leftovers
        for remaining in hoppers.values():
            if remaining:
                yield Batch(remaining)