import re
from src.knowledge.ekegusii_fewshot_corpus import FEWSHOT_CORPUS

class FewShotRetriever:
    @staticmethod
    def tokenize(text):
        # Normalize and split into lowercase words
        return set(re.findall(r'\b\w+\b', text.lower()))

    @staticmethod
    def get_similarity(set1, set2):
        # Jaccard similarity
        union = set1.union(set2)
        if not union:
            return 0.0
        return len(set1.intersection(set2)) / len(union)

    @classmethod
    def retrieve(cls, input_text, top_k=3):
        input_tokens = cls.tokenize(input_text)
        scored_corpus = []
        for example in FEWSHOT_CORPUS:
            example_tokens = cls.tokenize(example["English"])
            score = cls.get_similarity(input_tokens, example_tokens)
            scored_corpus.append((score, example))
        
        # Sort by score descending
        scored_corpus.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored_corpus[:top_k]]
