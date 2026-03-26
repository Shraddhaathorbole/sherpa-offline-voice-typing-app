"""
Post-processing pipeline applied to Whisper's raw transcript:

  1. Normalise whitespace
  2. Fast O(N) unified SymSpell pass (removes dupes, split fixes, and corrects)
  3. Apply sentence casing
"""

import logging
import os
import re
from typing import List

from symspellpy import SymSpell, Verbosity

logger = logging.getLogger(__name__)


class TextPostProcessor:
    def __init__(self):
        self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        self._load_dictionary()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, text: str, language: str = None) -> str:
        # Whisper BPE tokenizer may hallucinate invalid UTF-8 bytes for Marathi/Hindi, yielding ``.
        text = text.replace("\ufffd", "").replace("", "")
        
        # Whisper "small" heavily hallucinates the Tibetan 'ༀ' token and repeats it infinitely on silence.
        text = text.replace("ༀ", "")
        
        # Collapse any single character repeating more than 4 times in a row (typical hallucination loop)
        text = re.sub(r'(.)\1{4,}', r'\1', text)
        
        text = self._normalize_spaces(text)
        if not text:
            return ""

        # Symspell is English-only. Skip spelling correction if Marathi is selected.
        if language != "mr":
            text = self._fast_unified_spell_pass(text)
        else:
            text = self._remove_adjacent_duplicates(text)
            
        text = self._sentence_case(text)
        return text

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_dictionary(self) -> None:
        import symspellpy

        base_dir = os.path.dirname(symspellpy.__file__)
        dictionary_path = os.path.join(base_dir, "frequency_dictionary_en_82_765.txt")

        if not self.sym_spell.load_dictionary(dictionary_path, 0, 1):
            raise RuntimeError(
                "TextPostProcessor: could not load SymSpell frequency dictionary."
            )
        logger.info("TextPostProcessor: SymSpell dictionary loaded.")



    def _normalize_spaces(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def _remove_adjacent_duplicates(self, text: str) -> str:
        words = text.split()
        cleaned: List[str] = []
        for w in words:
            if not cleaned or cleaned[-1].lower() != w.lower():
                cleaned.append(w)
        return " ".join(cleaned)



    def _fast_unified_spell_pass(self, text: str) -> str:
        """
        An O(N) optimized unified pass that simultaneously identically handles:
        - Removing adjacent duplicated output (happens natively with pointers)
        - Rejoining split words 
        - Correcting spelling
        """
        words = text.split()
        if not words:
            return ""

        merged = []
        i = 0
        
        while i < len(words):
            word = words[i]
            
            # 1. Skip if it's identical to the immediately preceding appended word
            if merged and merged[-1].lower() == word.lower():
                i += 1
                continue
                
            # 2. Lookahead optimization for split-word fixing (e.g. "a" "bout" -> "about")
            if i < len(words) - 1:
                w_next = words[i + 1]
                combined = (word + w_next).lower()
                
                combined_sug = self.sym_spell.lookup(combined, Verbosity.CLOSEST, max_edit_distance=1)
                
                # If merging them forms a valid dictionary word, whereas apart they aren't both valid
                if combined_sug:
                    w1_valid = bool(self.sym_spell.lookup(word.lower(), Verbosity.CLOSEST, max_edit_distance=0))
                    w2_valid = bool(self.sym_spell.lookup(w_next.lower(), Verbosity.CLOSEST, max_edit_distance=0))
                    if not (w1_valid and w2_valid):
                        merged.append(combined_sug[0].term)
                        i += 2
                        continue
                        
            # 3. Handle the individual word
            sug = self.sym_spell.lookup(word.lower(), Verbosity.CLOSEST, max_edit_distance=2, include_unknown=True)
            best_guess = sug[0].term if sug else word
            
            # Word-splitting optimization ("appletree" -> "apple tree")
            if len(word) > 6:
                split_found = False
                for j in range(2, len(word) - 2):
                    left, right = word[:j], word[j:]
                    left_valid = self.sym_spell.lookup(left, Verbosity.CLOSEST, max_edit_distance=0)
                    right_valid = self.sym_spell.lookup(right, Verbosity.CLOSEST, max_edit_distance=0)
                    if left_valid and right_valid:
                        merged.append(left + " " + right)
                        split_found = True
                        break
                if split_found:
                    i += 1
                    continue

            merged.append(best_guess)
            i += 1

        return " ".join(merged)

    def _sentence_case(self, text: str) -> str:
        text = text.strip()
        return (text[0].upper() + text[1:]) if text else ""
