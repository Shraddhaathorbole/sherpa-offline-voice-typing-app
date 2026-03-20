import re
from deepmultilingualpunctuation import PunctuationModel


class TextPostProcessor:
    def __init__(self):
        self.model = PunctuationModel()

    def process(self, text: str) -> str:
        if not text:
            return ""

        text = self._basic_cleanup(text)
        text = self._restore_punctuation(text)
        text = self._grammar_cleanup(text)
        return text

    def _basic_cleanup(self, text: str) -> str:
        text = text.strip()

        # remove duplicated adjacent words
        words = text.split()
        cleaned = []
        for w in words:
            if not cleaned or cleaned[-1].lower() != w.lower():
                cleaned.append(w)

        text = " ".join(cleaned)

        # normalize spaces
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _restore_punctuation(self, text: str) -> str:
        try:
            punctuated = self.model.restore_punctuation(text)
            return punctuated.strip()
        except Exception:
            return text

    def _grammar_cleanup(self, text: str) -> str:
        if not text:
            return ""

        text = text.strip()

        # capitalize first letter
        text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()

        # small cleanup rules
        replacements = {
            " i ": " I ",
            " i'm ": " I'm ",
            " i'll ": " I'll ",
            " i've ": " I've ",
            " i'd ": " I'd ",
        }

        padded = f" {text} "
        for old, new in replacements.items():
            padded = padded.replace(old, new)
        text = padded.strip()

        # avoid doubled punctuation
        text = re.sub(r"([?.!,]){2,}", r"\1", text)

        # add ending punctuation if missing
        if text and text[-1] not in ".?!":
            text += "."

        return text