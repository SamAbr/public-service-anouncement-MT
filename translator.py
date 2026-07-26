try:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
except ImportError:
    torch = None
    AutoModelForSeq2SeqLM = None
    AutoTokenizer = None

from tqdm import tqdm
import math

class NLLBTranslator:
    def __init__(self, model_name="facebook/nllb-200-distilled-600M", src_lang="eng_Latn", tgt_lang="swh_Latn", batch_size=32):
        self.model_name = model_name
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.batch_size = batch_size
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if (torch and torch.cuda.is_available()) else "cpu"

    def load_model(self):
        if torch is None or AutoModelForSeq2SeqLM is None:
            raise ImportError("Required packages 'torch' and/or 'transformers' are not installed.")
            
        print(f"Loading local NLLB model '{self.model_name}' on device '{self.device}'...")
        # Note: AutoTokenizer will automatically download the necessary configuration
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, src_lang=self.src_lang)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
        print("Model and tokenizer loaded successfully.")

    def translate_batch(self, texts):
        """
        Translates a list of English texts into Swahili.
        """
        if not texts:
            return []

        if not self.model or not self.tokenizer:
            self.load_model()
            
        translated_texts = []
        num_batches = math.ceil(len(texts) / self.batch_size)
        
        for i in tqdm(range(num_batches), desc="Translating to Swahili"):
            batch_texts = texts[i * self.batch_size : (i + 1) * self.batch_size]
            
            # Tokenize inputs
            inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
            
            # Generate translation tokens
            with torch.no_grad():
                translated_tokens = self.model.generate(
                    **inputs,
                    forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(self.tgt_lang),
                    max_length=128,
                    num_beams=1  # Greedy search: ~5x speedup compared to default beam search (size 5)
                )
                
            # Decode tokens back into Swahili text
            batch_translations = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
            translated_texts.extend(batch_translations)
            
        return translated_texts
