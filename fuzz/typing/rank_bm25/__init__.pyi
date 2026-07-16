from collections.abc import Iterable, Sequence

class BM25Okapi:
    def __init__(
        self,
        corpus: Sequence[Sequence[str]],
        tokenizer: None = None,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
    ) -> None: ...
    def get_scores(self, query: Sequence[str]) -> Iterable[float]: ...
