import logging
import os
from collections import defaultdict
from typing import Any, Dict, Iterable, Iterator, List, Set
import json

import numpy as np
import torch
import torch.optim as optim
import tqdm
from allennlp.common import Registrable
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.common.util import END_SYMBOL, START_SYMBOL
from allennlp.data import Instance
from allennlp.data.dataset import Batch
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import MetadataField, SequenceLabelField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.tokenizers.word_splitter import SimpleWordSplitter
from overrides import overrides

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

torch.manual_seed(1)

@DatasetReader.register("unsupervised_translation")
class UnsupervisedTranslationDatasetsReader(DatasetReader):
    """
    """
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._undefined_lang_id = "xx"
        self._tokenizer = tokenizer or WordTokenizer(word_splitter=SimpleWordSplitter())
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._denoising_dataset_reader = DenoisingAutoencoderDatasetReader(tokenizer=tokenizer,
                                                                           token_indexers=token_indexers, lazy=lazy)
        self._backtranslation_dataset_reader = BacktranslationDatasetReader(tokenizer=tokenizer,
                                                                            token_indexers=token_indexers, lazy=lazy)
        self._parallel_dataset_reader = ParallelDatasetReader(source_tokenizer=tokenizer,
                                                              source_token_indexers=token_indexers, lazy=lazy)

        self._mingler = RoundRobinMingler(dataset_name_field="lang_pair", take_at_a_time=1)

    @overrides
    def _read(self, file_paths: Dict[str, str]):
        if type(file_paths) == str:  # if we ese allennlp evaluate, we pass the file paths dict in the form of a string
            file_paths = json.loads(file_paths)
        else:
            file_paths = dict(file_paths)

        datasets = {}
        for lang_code, path in file_paths.items():
            if len(lang_code.split('-')) == 1:
                lang_pair = lang_code + "-" + lang_code # 'en' becomes -> 'en-en' for consistancy. (denoising)
                datasets[lang_pair] = self._denoising_dataset_reader._read(path)

                lang_pair = self._undefined_lang_id + "-" + lang_code  # this means backtranslation examples
                datasets[lang_pair] = self._backtranslation_dataset_reader._read(path)

            elif len(lang_code.split('-')) == 2:
                lang_pair = lang_code
                datasets[lang_pair] = self._parallel_dataset_reader._read(path)

        return self._mingler.mingle(datasets=datasets)


    @overrides
    def text_to_instance(self, string: str, target_lang: str) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        """
        Used in predicition time
        """
        lang_pair = self._undefined_lang_id + '-' + target_lang
        tokenized_string = self._tokenizer.tokenize(string)
        string_field = TextField(tokenized_string, self._token_indexers)
        return Instance({self._mingler.dataset_name_field: MetadataField(lang_pair), 'source_tokens': string_field})

    def string_to_instance(self, string: str) -> Instance:
        """
        Used for backtranslation
        """
        tokenized_string = self._tokenizer.tokenize(string)
        string_field = TextField(tokenized_string, self._token_indexers)
        return Instance({'source_tokens': string_field})


class ParallelDatasetReader(DatasetReader):
    """
    """
    def __init__(self,
                 source_tokenizer: Tokenizer = None,
                 target_tokenizer: Tokenizer = None,
                 source_token_indexers: Dict[str, TokenIndexer] = None,
                 target_token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._source_tokenizer = source_tokenizer or WordTokenizer(word_splitter=SimpleWordSplitter())
        self._target_tokenizer = target_tokenizer or self._source_tokenizer
        self._source_token_indexers = source_token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._target_token_indexers = target_token_indexers or self._source_token_indexers

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line_num, line in enumerate(data_file):
                line = line.strip("\n")

                if not line:
                    continue

                line_parts = line.split('\t')
                if len(line_parts) != 2:
                    raise ConfigurationError("Invalid line format: %s (line number %d)" % (line, line_num + 1))
                source_sequence, target_sequence = line_parts
                yield self.text_to_instance(source_sequence, target_sequence)

    @overrides
    def text_to_instance(self, source_string: str, target_string: str = None) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        tokenized_source = self._source_tokenizer.tokenize(source_string)
        source_field = TextField(tokenized_source, self._source_token_indexers)
        if target_string is not None:
            tokenized_target = self._target_tokenizer.tokenize(target_string)
            target_field = TextField(tokenized_target, self._target_token_indexers)
            return Instance({"source_tokens": source_field, "target_tokens": target_field})
        else:
            return Instance({'source_tokens': source_field})


class DenoisingAutoencoderDatasetReader(DatasetReader):
    """
    """
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer(word_splitter=SimpleWordSplitter())
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line_num, line in enumerate(data_file):
                line = line.strip("\n")

                if not line:
                    continue

                yield self.text_to_instance(line)

    def _add_noise(self, sentence):
        # TODO: implement noising
        return sentence

    @overrides
    def text_to_instance(self, string: str) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        tokenized_string = self._tokenizer.tokenize(string)
        noised_string = self._add_noise(tokenized_string)
        string_field = TextField(tokenized_string, self._token_indexers)
        noised_string_filed = TextField(noised_string, self._token_indexers)
        return Instance({'source_tokens': noised_string_filed, 'target_tokens': string_field})


class BacktranslationDatasetReader(DatasetReader):
    """
    """
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer(word_splitter=SimpleWordSplitter())
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line_num, line in enumerate(data_file):
                line = line.strip("\n")

                if not line:
                    continue

                yield self.text_to_instance(line)

    @overrides
    def text_to_instance(self, string: str) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        tokenized_string = self._tokenizer.tokenize(string)
        string_field = TextField(tokenized_string, self._token_indexers)
        return Instance({'target_tokens': string_field})


class DatasetMingler(Registrable):
    """
    Our ``DataIterator`` class expects a single dataset;
    this is an abstract class for combining multiple datasets into one.
    You could imagine an alternate design where there is a
    ``MinglingDatasetReader`` that wraps multiple dataset readers,
    but then somehow you'd have to get it multiple file paths.
    """
    def mingle(self, datasets: Dict[str, Iterable[Instance]]) -> Iterable[Instance]:
        raise NotImplementedError


class RoundRobinMingler(DatasetMingler):
    """
    Cycle through datasets, ``take_at_time`` instances at a time.
    """
    def __init__(self,
                 dataset_name_field: str = "lang_pair",
                 take_at_a_time: int = 1) -> None:
        self.dataset_name_field = dataset_name_field
        self.take_at_a_time = take_at_a_time

    def mingle(self, datasets: Dict[str, Iterable[Instance]]) -> Iterable[Instance]:
        iterators = {name: iter(dataset) for name, dataset in datasets.items()}
        done: Set[str] = set()

        while iterators.keys() != done:
            for name, iterator in iterators.items():
                if name not in done:
                    try:
                        for _ in range(self.take_at_a_time):
                            instance = next(iterator)
                            instance.fields[self.dataset_name_field] = MetadataField(name)
                            yield instance
                    except StopIteration:
                        done.add(name)

