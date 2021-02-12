import warnings

from FrEIA.framework.sequence_inn import SequenceINN


class ReversibleSequential(SequenceINN):
    def __init__(self, *dims: int):
        warnings.warn("ReversibleSequential is deprecated in favour of "
                      "SequenceINN. It will be removed in a future version "
                      "of FrEIA.",
                      DeprecationWarning)
        super().__init__(*dims)
