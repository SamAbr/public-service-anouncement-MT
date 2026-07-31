import re
import numpy as np

try:
    from sentence_transformers import SentenceTransformer, util
    import torch
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

class ValidationEngine:
    def __init__(self, min_words=12, max_words=40):
        self.min_words = min_words
        self.max_words = max_words
        self.embedding_model = None
        self.seen_embeddings = []
        
        if HAS_SENTENCE_TRANSFORMERS:
            try:
                # Load a lightweight, fast semantic similarity model
                self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception as e:
                print(f"Warning: Failed to load SentenceTransformer: {e}. Falling back to n-gram overlap.")

    def validate(self, text: str, domain: str = None) -> tuple:
        """
        Runs Realism Validator rules on the generated PSA.
        Returns (is_valid, reason)
        """
        if not text or not isinstance(text, str):
            return False, "Empty or invalid type"
            
        text = text.strip()
        words = text.split()
        word_count = len(words)
        
        # 1. Word count constraint (12 to 40 words)
        if word_count < self.min_words or word_count > self.max_words:
            return False, f"Word count ({word_count}) is outside range [{self.min_words}, {self.max_words}]"
            
        # 2. Case and punctuation
        if not text[0].isupper():
            return False, "Sentence must start with a capital letter"
            
        if not text.endswith(('.', '!', '?')):
            return False, "Sentence must end with proper punctuation (. ! ?)"
            
        # 3. Prevent duplicated modifiers / words
        # E.g. "immediately ... immediately" or "online ... online"
        lower_text = text.lower()
        duplicate_words = ["immediately", "promptly", "urgently", "online", "portal", "official"]
        for dw in duplicate_words:
            if len(re.findall(rf"\b{dw}\b", lower_text)) > 1:
                return False, f"Duplicate modifier detected: '{dw}'"
                
        # 4. Refuse informal terms
        informal_words = ["hey", "yeah", "gonna", "wanna", "lol", "brb", "cool", "guys"]
        for word in informal_words:
            if re.search(rf"\b{word}\b", lower_text):
                return False, f"Contains informal language: '{word}'"
                
        # 5. Clean spacing
        if "  " in text:
            return False, "Contains consecutive double spaces"
            
        return True, "Valid"

    def is_semantically_unique(self, text: str, threshold: float = 0.92) -> bool:
        """
        Computes semantic similarity using SentenceTransformer.
        Falls back to token Jaccard similarity if SentenceTransformer is unavailable.
        """
        if not text:
            return False
            
        if HAS_SENTENCE_TRANSFORMERS and self.embedding_model is not None:
            try:
                new_emb = self.embedding_model.encode(text, convert_to_tensor=True)
                if not self.seen_embeddings:
                    self.seen_embeddings.append(new_emb)
                    return True
                    
                # Compute cosine similarities against all previous sentences
                # seen_embeddings is a list of tensors, stack them for parallel computation
                stacked = torch.stack(self.seen_embeddings)
                similarities = util.cos_sim(new_emb, stacked)[0]
                max_sim = torch.max(similarities).item()
                
                if max_sim > threshold:
                    return False
                
                # Append to our list of cache
                self.seen_embeddings.append(new_emb)
                return True
            except Exception as e:
                # Fallback to Jaccard if tensor operation fails
                pass
                
        # Fallback: Character/Token Jaccard Similarity
        return self._check_jaccard_uniqueness(text, threshold)

    def _check_jaccard_uniqueness(self, text: str, threshold: float) -> bool:
        """Fallback word n-gram Jaccard check if sentence embeddings are disabled"""
        if not hasattr(self, "_seen_token_sets"):
            self._seen_token_sets = []
            
        words = set(text.lower().split())
        for existing in self._seen_token_sets:
            intersection = len(words.intersection(existing))
            union = len(words.union(existing))
            if union == 0:
                continue
            jaccard = intersection / union
            if jaccard > (threshold - 0.07):  # Adjust threshold (e.g., 0.85 for 0.92 limit) to prevent over-filtering
                return False
                
        self._seen_token_sets.append(words)
        return True

    def calculate_chrf(self, ref: str, hyp: str, n=6, beta=3.0) -> float:
        """
        Pure Python fallback implementation of chrF score (character n-gram F-score).
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

    def validate_back_translation(self, original_text: str, back_translated_text: str, min_chrf=0.25) -> tuple:
        """
        Compares back-translated English to the original English.
        Returns (is_valid, chrF_score)
        """
        score = self.calculate_chrf(original_text, back_translated_text)
        return score >= min_chrf, score
