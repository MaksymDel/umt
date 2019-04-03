from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor


class UnsupervisedTranslationPredictor(Predictor):
    """
    Predictor for sequence to sequence models, including
    :class:`~allennlp.models.encoder_decoder.simple_seq2seq` and
    :class:`~allennlp.models.encoder_decoder.copynet_seq2seq`.
    """

    def predict(self, source: str, lang: str) -> JsonDict:
        return self.predict_json({"lang": lang, "source" : source})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"lang", "source": "..."}``.
        """
        source = json_dict["source"]
        lang = json_dict["lang"]
        return self._dataset_reader.text_to_instance(source, lang)