"""Reserved evidence overlap, without shared question/candidate schemas.

This catches literal/normalized excerpts and ancestry-linked groups. It does
not prove semantic independence from paraphrases or unlisted private data.
"""

from .decontam import _CJK, Decontaminator, _state_texts, _words
from .decontam import POLICY as PUBLIC_POLICY

POLICY = {
    **{key: value for key, value in PUBLIC_POLICY.items() if key not in {"version", "scope"}},
    "version": 1,
    "scope": "reserved state evidence only; complete source-group exclusion",
    "short_word_match": "contiguous normalized tokens inside full document",
    "wrapped_cjk_overlap": "12 consecutive normalized CJK characters",
}


class ReservedEvidenceBlocker(Decontaminator):
    def __init__(self, samples):
        samples = tuple(samples)
        super().__init__(
            [text for sample in samples for text in _state_texts(sample.state)],
            n=POLICY["word_ngram"],
            min_exact_words=POLICY["min_exact_words"],
        )
        self.short_grams = {}
        self.wrapped_cjk_grams = set()
        for sample in samples:
            for text in _state_texts(sample.state):
                self.wrapped_cjk_grams.update(self._wrapped_cjk(text))
        for text in self.exact:
            words = tuple(text.split())
            self.short_grams.setdefault(len(words), set()).add(words)

    def text_hit(self, text):
        if super().text_hit(text) or self._wrapped_cjk(text) & self.wrapped_cjk_grams:
            return True
        words = _words(text)
        return any(
            tuple(words[i : i + width]) in grams
            for width, grams in self.short_grams.items()
            for i in range(len(words) - width + 1)
        )

    def state_hit(self, sample):
        return any(self.text_hit(text) for text in _state_texts(sample.state))

    @staticmethod
    def _wrapped_cjk(text):
        normalized = "".join(_words(text))
        width = POLICY["min_exact_cjk_characters"]
        return {
            normalized[i : i + width]
            for i in range(len(normalized) - width + 1)
            if len(_CJK.findall(normalized[i : i + width])) == width
        }
