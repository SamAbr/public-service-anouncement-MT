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

    def calculate_chrf(self, ref, hyp, n=6, beta=3.0):
        """
        Pure Python fallback implementation of chrF score (character n-gram F-score).
        Matches the NLTK chrF metric implementation.
        """
        ref_clean = ref.replace(" ", "")
        hyp_clean = hyp.replace(" ", "")
        if not ref_clean or not hyp_clean:
            return 0.0
            
        def get_ngrams(s, length):
            return [s[i:i+length] for i in range(len(s) - length + 1)]
            
        total_precision = 0.0
        total_recall = 0.0
        valid_ns = 0
        
        for i in range(1, n + 1):
            ref_ngrams = get_ngrams(ref_clean, i)
            hyp_ngrams = get_ngrams(hyp_clean, i)
            if not ref_ngrams or not hyp_ngrams:
                continue
            
            ref_counts = {}
            for ng in ref_ngrams:
                ref_counts[ng] = ref_counts.get(ng, 0) + 1
                
            matches = 0
            for ng in hyp_ngrams:
                if ref_counts.get(ng, 0) > 0:
                    matches += 1
                    ref_counts[ng] -= 1
                    
            precision = matches / len(hyp_ngrams)
            recall = matches / len(ref_ngrams)
            total_precision += precision
            total_recall += recall
            valid_ns += 1
            
        if valid_ns == 0:
            return 0.0
            
        avg_p = total_precision / valid_ns
        avg_r = total_recall / valid_ns
        
        if avg_p + avg_r == 0:
            return 0.0
            
        beta_sq = beta ** 2
        f_score = (1 + beta_sq) * (avg_p * avg_r) / ((beta_sq * avg_p) + avg_r)
        return f_score

    def validate_back_translation(self, original_text, back_translated_text, min_chrf=0.25):
        """
        Compares back-translated English to the original English.
        Returns (is_valid, chrF_score)
        """
        score = self.calculate_chrf(original_text, back_translated_text)
        return score >= min_chrf, score
