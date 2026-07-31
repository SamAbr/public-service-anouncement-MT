try:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
except ImportError:
    torch = None
    AutoModelForSeq2SeqLM = None
    AutoTokenizer = None

import gc
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

    def load_model(self, model_name=None):
        if torch is None or AutoModelForSeq2SeqLM is None:
            raise ImportError("Required packages 'torch' and/or 'transformers' are not installed.")
            
        m_name = model_name if model_name else self.model_name
        print(f"Loading local NLLB model '{m_name}' on device '{self.device}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(m_name, src_lang=self.src_lang)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            m_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        print("Model and tokenizer loaded successfully.")

    def unload_model(self):
        print("Unloading model and clearing cache...")
        self.model = None
        self.tokenizer = None
        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    def translate_batch(self, texts, tgt_lang=None):
        """
        Translates a list of English texts into the target language.
        """
        if not texts:
            return []

        target = tgt_lang if tgt_lang else self.tgt_lang

        if not self.model or not self.tokenizer:
            self.load_model()
            
        translated_texts = []
        num_batches = math.ceil(len(texts) / self.batch_size)
        tgt_lang_id = self.tokenizer.convert_tokens_to_ids(target)
        
        for i in tqdm(range(num_batches), desc=f"Translating to {target}"):
            batch_texts = texts[i * self.batch_size : (i + 1) * self.batch_size]
            
            # Tokenize inputs
            inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
            
            # Generate translation tokens
            with torch.no_grad():
                translated_tokens = self.model.generate(
                    **inputs,
                    forced_bos_token_id=tgt_lang_id,
                    max_length=64,
                    num_beams=1  # Greedy search
                )
                
            # Decode tokens
            batch_translations = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
            translated_texts.extend(batch_translations)
            
        return translated_texts
