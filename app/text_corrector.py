"""
Lightweight spell-correction helper (used standalone or as a pre-processing
step before punctuation restoration).
Backed by SymSpell for O(1) average-case lookup.
"""

import logging
import os
import re
from typing import List

from symspellpy import SymSpell, Verbosity

logger = logging.getLogger(__name__)


class TextCorrector:
    def __init__(self, max_dictionary_edit_distance: int = 2, prefix_length: int = 7):
        self.sym_spell = SymSpell(
            max_dictionary_edit_distance=max_dictionary_edit_distance,
            prefix_length=prefix_length,
        )
        self._load_dictionary()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def correct_text(self, text: str) -> str:
        text = self._normalize_spaces(text)
        if not text:
            return ""

        words = text.split()
        corrected_words: List[str] = [self._correct_word(w) for w in words]

        corrected = " ".join(corrected_words)
        corrected = self._merge_split_words(corrected)
        corrected = self._dedupe_adjacent_words(corrected)
        corrected = self._sentence_case(corrected)
        return corrected

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_dictionary(self) -> None:
        import symspellpy

        base_dir = os.path.dirname(symspellpy.__file__)
        dictionary_path = os.path.join(base_dir, "frequency_dictionary_en_82_765.txt")

        if not os.path.exists(dictionary_path):
            raise FileNotFoundError(
                "SymSpell frequency dictionary not found. "
                "Re-install symspellpy or provide a custom dictionary path."
            )

        loaded = self.sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
        if not loaded:
            raise RuntimeError("SymSpell failed to load the frequency dictionary.")

        logger.info("TextCorrector: dictionary loaded from '%s'.", dictionary_path)

    def _correct_word(self, word: str) -> str:
        prefix, core, suffix = self._split_punctuation(word)
        if not core or len(core) <= 1:
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
        """Merge common ASR split-word artefacts (e.g. 'af ter' → 'after')."""
        words = text.split()
        if len(words) < 2:
            return text

        merged: List[str] = []
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
                # Only merge short fragments to avoid false positives
                if len(words[i]) <= 4 and len(words[i + 1]) <= 5:
                    if suggestions:
                        top = suggestions[0]
                        if top.term == combined.lower() and top.distance <= 1:
                            merged.append(combined.lower())
                            i += 2
                            continue
                        if top.term != combined.lower():
                            merged.append(top.term)
                            i += 2
                            continue

            merged.append(words[i])
            i += 1

        return " ".join(merged)

    def _dedupe_adjacent_words(self, text: str) -> str:
        words = text.split()
        cleaned: List[str] = []
        for w in words:
            if not cleaned or cleaned[-1].lower() != w.lower():
                cleaned.append(w)
        return " ".join(cleaned)

    def _sentence_case(self, text: str) -> str:
        text = text.strip()
        return (text[0].upper() + text[1:]) if text else ""

    def _normalize_spaces(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _split_punctuation(word: str):
        match = re.match(r"^([^A-Za-z0-9]*)([A-Za-z0-9']+)([^A-Za-z0-9]*)$", word)
        if not match:
            return "", word, ""
        return match.group(1), match.group(2), match.group(3)
