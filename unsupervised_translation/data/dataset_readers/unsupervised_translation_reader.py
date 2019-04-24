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

# TODO: find a better place for this
golden_tokenizer = WordTokenizer(word_splitter=SimpleWordSplitter())
golden_token_indexers = {"golden_tokens": SingleIdTokenIndexer(namespace="tokens")}


def string_to_fields(string: str, tokenizer: Tokenizer, token_indexers: Dict[str, TokenIndexer]):
    tokenized_string = tokenizer.tokenize(string)
    tokenized_string.insert(0, Token(END_SYMBOL))
    field = TextField(tokenized_string, token_indexers)

    # TODO: always use single id token indexer and tokenizer default/bpe cause we will have bert/elmo passed to main str
    tokenized_golden_string = golden_tokenizer.tokenize(string)
    tokenized_golden_string.append(Token(END_SYMBOL))  # with eos at the end for loss compute
    field_golden = TextField(tokenized_golden_string, golden_token_indexers)

    return field, field_golden


@DatasetReader.register("unsupervised_translation")
class UnsupervisedTranslationDatasetsReader(DatasetReader):
    """
    """
    def __init__(self,
                 langs_list: List[str],
                 ae_steps: List[str] = None,
                 bt_steps: List[str] = None,
                 para_steps: List[str] = None,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False
                 ) -> None:
        super().__init__(lazy)
        self._undefined_lang_id = "xx"
        self._tokenizer = tokenizer or WordTokenizer(word_splitter=SimpleWordSplitter())
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._denoising_dataset_reader = ParallelDatasetReader(lang1_tokenizer=tokenizer,
                                                               lang1_token_indexers=token_indexers,
                                                               lazy=lazy, denoising=True)
        self._backtranslation_dataset_reader = BacktranslationDatasetReader(tokenizer=tokenizer,
                                                                            token_indexers=token_indexers, lazy=lazy)
        self._parallel_dataset_reader = ParallelDatasetReader(lang1_tokenizer=tokenizer,
                                                              lang1_token_indexers=token_indexers, lazy=lazy)

        self._mingler = RoundRobinMingler(dataset_name_field="lang_pair", take_at_a_time=1)

        self._langs_list = langs_list
        self._ae_steps = ae_steps
        self._bt_steps = bt_steps
        self._para_steps = para_steps

    @overrides
    def _read(self, file_path):
        dir_path = file_path
        filenames = os.listdir(dir_path)
        filenames = [dir_path+f for f in filenames]
        print(filenames)

        para_filenames = []
        for para_step in self._para_steps:
            for filename in filenames:
                if filename.endswith('.' + para_step):
                    para_filenames.append(filename)
        assert len(para_filenames) == len(self._para_steps)

        ae_filenames = []
        for ae_step in self._ae_steps:
            for filename in filenames:
                if filename.endswith('.' + ae_step):
                    ae_filenames.append(filename)
        assert len(ae_filenames) == len(self._ae_steps)

        bt_filenames = []
        for bt_step in self._bt_steps:
            for filename in filenames:
                if filename.endswith("." + bt_step):
                    bt_filenames.append(filename)
        assert len(bt_filenames) == len(self._bt_steps)

        datasets = {}
        for lang_pair_code, filename in zip(self._para_steps, para_filenames):
            lang_pair = lang_pair_code
            datasets[lang_pair] = self._parallel_dataset_reader._read(filename)

        for lang_code, filename in zip(self._ae_steps, ae_filenames):
            lang_pair = lang_code + "-" + lang_code  # 'en' becomes -> 'en-en' for consistancy. (denoising)
            datasets[lang_pair] = self._denoising_dataset_reader._read(filename)

        for lang_code, filename in zip(self._bt_steps, bt_filenames):
            lang_pair = self._undefined_lang_id + "-" + lang_code  # this means backtranslation examples
            datasets[lang_pair] = self._backtranslation_dataset_reader._read(filename)

        return self._mingler.mingle(datasets=datasets)




    @overrides
    def text_to_instance(self, string: str, lang2_lang: str) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        """
        Used in predicition time
        """
        lang_pair = self._undefined_lang_id + '-' + lang2_lang
        tokenized_string = self._tokenizer.tokenize(string)
        string_field = TextField(tokenized_string, self._token_indexers)
        return Instance({self._mingler.dataset_name_field: MetadataField(lang_pair), 'lang1_tokens': string_field})

    def string_to_instance(self, string: str) -> Instance:
        """
        Used for backtranslation
        """
        string_field, golden_field = string_to_fields(string, self._tokenizer, self._token_indexers)
        return Instance({'tokens': string_field, 'golden': golden_field})


class ParallelDatasetReader(DatasetReader):
    """
    """
    def __init__(self,
                 lang1_tokenizer: Tokenizer = None,
                 lang2_tokenizer: Tokenizer = None,
                 lang1_token_indexers: Dict[str, TokenIndexer] = None,
                 lang2_token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False,
                 denoising=False) -> None:
        super().__init__(lazy)
        self._lang1_tokenizer = lang1_tokenizer or WordTokenizer(word_splitter=SimpleWordSplitter())
        self._lang2_tokenizer = lang2_tokenizer or self._lang1_tokenizer
        self._lang1_token_indexers = lang1_token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._lang2_token_indexers = lang2_token_indexers or self._lang1_token_indexers
        self._denoising = denoising

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line_num, line in enumerate(data_file):
                line = line.strip("\n")

                if not line:
                    continue

                line_parts = line.split('\t')
                if not self._denoising:
                    if len(line_parts) != 2:
                        raise ConfigurationError("Invalid line format: %s (line number %d)" % (line, line_num + 1))
                    lang1_sequence, lang2_sequence = line_parts
                    yield self.text_to_instance(lang1_sequence, lang2_sequence)
                else:
                    if len(line_parts) != 1:
                        raise ConfigurationError("Invalid line format: %s (line number %d)" % (line, line_num + 1))
                    string = line[0]
                    noised = self._add_noise(string)
                    yield self.text_to_instance(noised, string)


    @overrides
    def text_to_instance(self, lang1_string: str, lang2_string: str = None) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        lang1_field, lang1_golden = string_to_fields(lang1_string, self._lang1_tokenizer, self._lang1_token_indexers)
        if lang2_string is not None:
            lang2_field, lang2_golden = string_to_fields(lang2_string, self._lang2_tokenizer, self._lang2_token_indexers)
            return Instance({"lang1_tokens": lang1_field, "lang2_tokens": lang2_field,
                             'lang1_golden': lang1_golden, "lang2_golden": lang2_golden})
        else:
            return Instance({'lang1_tokens': lang1_field, 'lang1_golden': lang1_golden})

    def _add_noise(self, sentence: str):
        return sentence


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
        lang2_field, lang2_golden = string_to_fields(string, self._tokenizer, self._token_indexers)
        return Instance({'lang2_tokens': lang2_field, 'lang2_golden': lang2_golden})


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

