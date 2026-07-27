import re

class ValidationEngine:
    def __init__(self, min_words=25, max_words=60):
        self.min_words = min_words
        self.max_words = max_words

    def validate(self, text, domain=None):
        """
        Validates the generated PSA text.
        Returns a tuple: (is_valid, reason)
        """
        if not text or not isinstance(text, str):
            return False, "Empty or invalid type"
            
        text = text.strip()
        if len(text) == 0:
            return False, "Text is empty"
            
        # Word count validation
        words = text.split()
        word_count = len(words)
        if word_count < self.min_words or word_count > self.max_words:
            return False, f"Word count ({word_count}) is outside range [{self.min_words}, {self.max_words}]"
            
        # Formal sentence validation
        if not text[0].isupper():
            return False, "Sentence must start with a capital letter"
            
        if not text.endswith(('.', '!', '?')):
            return False, "Sentence must end with proper punctuation (. ! ?)"
            
        # Tone / quality rules: reject informal or colloquial English words
        informal_words = ["hey", "yeah", "gonna", "wanna", "lol", "brb", "cool", "guys"]
        for word in informal_words:
            if re.search(rf"\b{word}\b", text.lower()):
                return False, f"Contains informal language: '{word}'"
                
        return True, "Valid"
