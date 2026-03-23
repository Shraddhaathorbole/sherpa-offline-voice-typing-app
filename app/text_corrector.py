import os
import re
from typing import List

from symspellpy import SymSpell, Verbosity


class TextCorrector:
    def __init__(self, max_dictionary_edit_distance: int = 2, prefix_length: int = 7):
        self.sym_spell = SymSpell(
            max_dictionary_edit_distance=max_dictionary_edit_distance,
            prefix_length=prefix_length,
        )
        self._load_dictionary()

    def _load_dictionary(self):
        # Try built-in frequency dictionary from symspellpy package
        base_dir = os.path.dirname(__import__("symspellpy").__file__)
        dictionary_path = os.path.join(base_dir, "frequency_dictionary_en_82_765.txt")

        if not os.path.exists(dictionary_path):
            raise FileNotFoundError(
                "SymSpell dictionary not found. Reinstall symspellpy or provide a custom dictionary."
            )

        loaded = self.sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
        if not loaded:
            raise RuntimeError("Failed to load SymSpell dictionary.")

    def correct_text(self, text: str) -> str:
        text = self._normalize_spaces(text)
        if not text:
            return ""

        words = text.split()
        corrected_words: List[str] = []

        for word in words:
            corrected_words.append(self._correct_word(word))

        corrected = " ".join(corrected_words)
        corrected = self._merge_split_words(corrected)
        corrected = self._dedupe_adjacent_words(corrected)
        corrected = self._sentence_case(corrected)
        return corrected

    def _correct_word(self, word: str) -> str:
        # keep punctuation attached separately
        prefix, core, suffix = self._split_punctuation(word)
        if not core:
            return word

        # don't over-correct very short tokens too aggressively
        if len(core) <= 1:
            return word

        suggestions = self.sym_spell.lookup(
            core.lower(),
            Verbosity.CLOSEST,
            max_edit_distance=2,
            include_unknown=True,
        )

        best = suggestions[0].term if suggestions else core.lower()
        return f"{prefix}{best}{suffix}"

    def _merge_split_words(self, text: str) -> str:
        """
        Merge accidental split words like:
        'hu llo' -> 'hello'
        'after noon' stays if valid as two words? We only merge when merged form looks better.
        """
        words = text.split()
        if len(words) < 2:
            return text

        merged = []
        i = 0
        while i < len(words):
            if i < len(words) - 1:
                combined = words[i] + words[i + 1]
                suggestion = self.sym_spell.lookup(
                    combined.lower(),
                    Verbosity.CLOSEST,
                    max_edit_distance=2,
                    include_unknown=True,
                )

                # merge only if combined correction is a known good word
                if suggestion and suggestion[0].term != combined.lower():
                    # only merge when both pieces are short-ish, typical ASR split pattern
                    if len(words[i]) <= 4 and len(words[i + 1]) <= 5:
                        merged.append(suggestion[0].term)
                        i += 2
                        continue

                # also accept exact known merged word
                if suggestion and suggestion[0].term == combined.lower() and suggestion[0].distance <= 1:
                    if len(words[i]) <= 4 and len(words[i + 1]) <= 5:
                        merged.append(combined.lower())
                        i += 2
                        continue

            merged.append(words[i])
            i += 1

        return " ".join(merged)

    def _dedupe_adjacent_words(self, text: str) -> str:
        words = text.split()
        cleaned = []
        for w in words:
            if not cleaned or cleaned[-1].lower() != w.lower():
                cleaned.append(w)
        return " ".join(cleaned)

    def _sentence_case(self, text: str) -> str:
        text = text.strip()
        if not text:
            return ""
        return text[0].upper() + text[1:]

    def _normalize_spaces(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def _split_punctuation(self, word: str):
        match = re.match(r"^([^A-Za-z0-9]*)([A-Za-z0-9']+)([^A-Za-z0-9]*)$", word)
        if not match:
            return "", word, ""
        return match.group(1), match.group(2), match.group(3)