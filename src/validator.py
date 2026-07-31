import re
import numpy as np

try:
    from sentence_transformers import SentenceTransformer, util
    import torch
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

class ValidationEngine:
    def __init__(self, min_words=10, max_words=25):
        self.min_words = min_words
        self.max_words = max_words
        self.embedding_model = None
        self.seen_embeddings = None
        self.seen_count = 0
        
        if HAS_SENTENCE_TRANSFORMERS:
            try:
                # Load a lightweight, fast semantic similarity model
                self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception as e:
                print(f"Warning: Failed to load SentenceTransformer: {e}. Falling back to n-gram overlap.")

    def validate(self, text: str, domain: str = None) -> tuple:
        """
        Runs PSA Framework Decision Tree and Realism rules on the generated PSA.
        Returns (is_valid, reason)
        """
        if not text or not isinstance(text, str):
            return False, "Empty or invalid type"
            
        text = text.strip()
        words = text.split()
        word_count = len(words)
        
        # 1. Word count constraint (10 to 25 words)
        if word_count < self.min_words or word_count > self.max_words:
            return False, f"Word count ({word_count}) is outside range [{self.min_words}, {self.max_words}]"
            
        # 2. Case and punctuation
        if not text[0].isupper():
            return False, "Sentence must start with a capital letter"
            
        if not text.endswith(('.', '!', '?')):
            return False, "Sentence must end with proper punctuation (. ! ?)"
            
        # 3. Ensure exactly one sentence (excluding standard abbreviation dots like KNEC, TSC, etc.)
        # Split by terminal punctuation and filter out empty strings
        clean_punc_text = re.sub(r'\b[A-Za-z]\.', '', text)  # remove initial dots like J. or Co.
        sentences = [s.strip() for s in re.split(r'[.!?]', clean_punc_text) if s.strip()]
        if len(sentences) != 1:
            return False, f"Must contain exactly one sentence. Found {len(sentences)}: {sentences}"

        lower_text = text.lower()

        # 4. PSA Framework Decision Tree Rule 1: Must tell the public to take action, avoid something, or be alert
        psa_keywords = ["advise", "urge", "warn", "remind", "request", "inform", "deadline", "alert", "verify", "submit", "report", "avoid", "comply", "register", "apply", "check", "secure"]
        if not any(kw in lower_text for kw in psa_keywords):
            return False, "Does not contain core PSA action/advisory directive keywords"

        # 5. PSA Framework Decision Tree Rule 2: Reject Press Release Style (events, launches, CS statements)
        press_release_keywords = ["launch", "inaugurated", "cabinet secretary", "official visit", "media invited", "press release", "meeting today", "announced that", "held a meeting", "CS "]
        for prw in press_release_keywords:
            if prw in lower_text:
                return False, f"Contains Press Release style keyword/content: '{prw}'"

        # 6. PSA Framework Decision Tree Rule 3: Reject Gazette / Legal Notice Style (legal/administrative Notice)
        gazette_keywords = ["gazette notice", "tender no", "pursuant to", "in exercise of the powers", "hereby notifies", "failure to comply", "hereby instructs"]
        for gzw in gazette_keywords:
            if gzw in lower_text:
                return False, f"Contains Gazette Notice style keyword/content: '{gzw}'"

        # 7. Prevent duplicated modifiers / words
        duplicate_words = ["immediately", "promptly", "urgently", "online", "portal", "official"]
        for dw in duplicate_words:
            if len(re.findall(rf"\b{dw}\b", lower_text)) > 1:
                return False, f"Duplicate modifier detected: '{dw}'"
                
        # 8. Refuse informal terms
        informal_words = ["hey", "yeah", "gonna", "wanna", "lol", "brb", "cool", "guys"]
        for word in informal_words:
            if re.search(rf"\b{word}\b", lower_text):
                return False, f"Contains informal language: '{word}'"
                
        # 9. Clean spacing
        if "  " in text:
            return False, "Contains consecutive double spaces"
            
        return True, "Valid"

    def is_semantically_unique(self, text: str, threshold: float = 0.92) -> bool:
        """
        Computes semantic similarity using SentenceTransformer with pre-allocated numpy BLAS dot product.
        Falls back to token Jaccard similarity if SentenceTransformer is unavailable.
        """
        if not text:
            return False
            
        if HAS_SENTENCE_TRANSFORMERS and self.embedding_model is not None:
            try:
                # encode returning L2-normalized numpy array
                new_emb = self.embedding_model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
                
                if self.seen_embeddings is None:
                    # Pre-allocate large matrix for up to 55,000 embeddings (size of all-MiniLM-L6-v2 is 384)
                    self.seen_embeddings = np.zeros((55000, 384), dtype=np.float32)
                    self.seen_embeddings[0] = new_emb
                    self.seen_count = 1
                    return True
                    
                # Compute cosine similarities (dot product since normalized)
                similarities = np.dot(self.seen_embeddings[:self.seen_count], new_emb)
                max_sim = np.max(similarities)
                
                if max_sim > threshold:
                    return False
                
                # Append to our pre-allocated array
                if self.seen_count < len(self.seen_embeddings):
                    self.seen_embeddings[self.seen_count] = new_emb
                    self.seen_count += 1
                return True
            except Exception as e:
                # Fallback to Jaccard if operation fails
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
