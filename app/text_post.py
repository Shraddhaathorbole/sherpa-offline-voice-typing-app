import re
from deepmultilingualpunctuation import PunctuationModel
from symspellpy import SymSpell, Verbosity


class TextPostProcessor:
    def __init__(self):
        self.punct = PunctuationModel()
        self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        self._load_dictionary()

    def _load_dictionary(self):
        import os
        import symspellpy
        base_dir = os.path.dirname(symspellpy.__file__)
        dictionary_path = os.path.join(base_dir, "frequency_dictionary_en_82_765.txt")
        if not self.sym_spell.load_dictionary(dictionary_path, 0, 1):
            raise RuntimeError("Could not load SymSpell dictionary")

    def process(self, text: str) -> str:
        text = self._normalize_spaces(text)
        if not text:
            return ""

        text = self._fix_split_words(text)
        text = self._spell_correct(text)
        text = self._remove_adjacent_duplicates(text)
        text = self._restore_punctuation(text)
        text = self._sentence_case(text)
        return text

    def _normalize_spaces(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def _fix_split_words(self, text: str) -> str:
        words = text.split()
        if len(words) < 2:
            return text

        merged = []
        i = 0
        while i < len(words):
            if i < len(words) - 1:
                combined = words[i] + words[i + 1]
                suggestions = self.sym_spell.lookup(
                    combined.lower(),
                    Verbosity.CLOSEST,
                    max_edit_distance=2,
                    include_unknown=True,
                )
                if suggestions and suggestions[0].term != combined.lower():
                    if len(words[i]) <= 4 and len(words[i + 1]) <= 5:
                        merged.append(suggestions[0].term)
                        i += 2
                        continue
                if suggestions and suggestions[0].term == combined.lower():
                    if len(words[i]) <= 4 and len(words[i + 1]) <= 5:
                        merged.append(combined.lower())
                        i += 2
                        continue

            merged.append(words[i])
            i += 1

        return " ".join(merged)

    def _spell_correct(self, text: str) -> str:
        corrected = []
        for word in text.split():
            suggestions = self.sym_spell.lookup(
                word.lower(),
                Verbosity.CLOSEST,
                max_edit_distance=2,
                include_unknown=True,
            )
            corrected.append(suggestions[0].term if suggestions else word)
        return " ".join(corrected)

    def _remove_adjacent_duplicates(self, text: str) -> str:
        words = text.split()
        cleaned = []
        for w in words:
            if not cleaned or cleaned[-1].lower() != w.lower():
                cleaned.append(w)
        return " ".join(cleaned)

    def _restore_punctuation(self, text: str) -> str:
        try:
            return self.punct.restore_punctuation(text)
        except Exception:
            return text

    def _sentence_case(self, text: str) -> str:
        text = text.strip()
        if not text:
            return ""
        return text[0].upper() + text[1:]