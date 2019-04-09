# get transformer parameters together
class ArgsStub:
    def __init__(self):
        pass


class DictStub:
    def __init__(self, num_tokens: int = None, pad: int = None, unk: int = None, eos: int = None):
        self._num_tokens = num_tokens
        self._pad = pad
        self._unk = unk
        self._eos = eos

    def pad(self):
        return self._pad

    def unk(self):
        return self._unk

    def eos(self):
        return self._eos

    def __len__(self):
        return self._num_tokens