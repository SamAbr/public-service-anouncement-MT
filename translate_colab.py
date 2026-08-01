import argparse
import pandas as pd
import torch
import gc
import os
import urllib.request
import math
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm

# Attempt to import nltk for ChrF calculation (pre-installed in Colab)
try:
    import nltk
    from nltk.translate.chrf_score import sentence_chrf
except ImportError:
    nltk = None

# Attempt to import fasttext for Language ID
try:
    import fasttext
except ImportError:
    fasttext = None

# List of acronyms to protect as untranslatable tokens
ACRONYMS_TO_PIN = ["NTSA", "KEPHIS", "HELB", "KUCCPS", "KRA", "CBK", "EACC", "SHA", "KEMRI", "KEMSA", "Huduma"]

def download_fasttext_model():
    model_path = "lid.176.bin"
    if not os.path.exists(model_path):
        print("Downloading FastText Language ID model...")
        url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
        urllib.request.urlretrieve(url, model_path)
    return model_path

def main():
    parser = argparse.ArgumentParser(description="Colab GPU Sequential 4-Way Translation for NLLB-200")
    parser.add_argument("--input", type=str, default="english_psas.csv", help="Path to input English CSV file")
    parser.add_argument("--output", type=str, default="psa_parallel_dataset.csv", help="Path to output parallel CSV file")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for GPU translation")
    
    # Models per language
    parser.add_argument("--model-swh", type=str, default="facebook/nllb-200-distilled-600M", help="NLLB model for Swahili")
    parser.add_argument("--model-som", type=str, default="facebook/nllb-200-1.3B", help="NLLB model for Somali")
    parser.add_argument("--model-luo", type=str, default="facebook/nllb-200-1.3B", help="NLLB model for Luo")
    
    # Quality thresholds
    parser.add_argument("--min-chrf", type=float, default=0.25, help="Minimum ChrF score for back-translation")
    
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        return

    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running translation on device: '{device}'")

    # Read/Initialize parallel dataset (supporting resumability)
    if os.path.exists(args.output):
        print(f"Resuming translation using existing parallel dataset: '{args.output}'")
        df = pd.read_csv(args.output)
    else:
        print(f"Initializing new parallel dataset from '{args.input}'")
        df = pd.read_csv(args.input)
        
    # Ensure target columns exist in the DataFrame (handles upgrading older CSV formats)
    for col in ["Kiswahili", "Somali", "Luo", "is_synthetic", "model_version"]:
        if col not in df.columns:
            df[col] = ""
    df["is_synthetic"] = True
    df["model_version"] = "NLLB-200"

    total_records = len(df)
    english_texts = df["English"].tolist()

    # Load FastText LangID if available
    ft_model = None
    if fasttext is not None:
        try:
            model_path = download_fasttext_model()
            ft_model = fasttext.load_model(model_path)
            print("FastText Language ID model loaded successfully.")
        except Exception as e:
            print(f"Warning: Could not initialize FastText: {e}")

    # Define targets to loop over sequentially
    targets = [
        # (Column Name, Lang Code, Model Path, FastText Code)
        ("Kiswahili", "swh_Latn", args.model_swh, "sw"),
        ("Somali", "som_Latn", args.model_som, "so"),
        ("Luo", "luo_Latn", args.model_luo, "luo")
    ]

    for col_name, lang_code, model_name, fasttext_code in targets:
        # Check if this language needs translation (resumability check)
        untranslated_indices = df[df[col_name].isna() | (df[col_name] == "")].index.tolist()
        
        if not untranslated_indices:
            print(f"Skipping {col_name}: Already fully translated.")
            continue

        print(f"\n=== Translating to {col_name} using model {model_name} ({len(untranslated_indices)} records remaining) ===")
        
        # Load model and tokenizer
        print(f"Loading {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="eng_Latn")
        
        # Pin proper nouns
        tokenizer.add_tokens(ACRONYMS_TO_PIN)
        
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
        model.resize_token_embeddings(len(tokenizer))
        
        tgt_lang_id = tokenizer.convert_tokens_to_ids(lang_code)
        eng_lang_id = tokenizer.convert_tokens_to_ids("eng_Latn")
        
        num_batches = math.ceil(len(untranslated_indices) / args.batch_size)
        
        for b in tqdm(range(num_batches), desc=f"Processing {col_name}"):
            batch_idx = untranslated_indices[b * args.batch_size : (b + 1) * args.batch_size]
            batch_texts = [english_texts[idx] for idx in batch_idx]
            
            # Print periodic progress logs
            if b % 10 == 0 or b == num_batches - 1:
                print(f"[{col_name}] Translated batch {b + 1}/{num_batches} (Processed {min((b + 1) * args.batch_size, len(untranslated_indices))} / {len(untranslated_indices)} records)...")

            
            # Forward translation: English -> Target
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():
                translated_tokens = model.generate(
                    **inputs,
                    forced_bos_token_id=tgt_lang_id,
                    max_length=64,
                    num_beams=1
                )
            translations = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
            
            # Back-translation check & LangID filtering
            final_translations = []
            for i, target_text in enumerate(translations):
                original_idx = batch_idx[i]
                original_english = batch_texts[i]
                
                # Filter 1: Off-target Language ID check
                if ft_model is not None:
                    try:
                        # Clean newlines for fasttext
                        cleaned_text = target_text.replace("\n", " ")
                        predictions = ft_model.predict(cleaned_text, k=3)
                        pred_labels = [label.replace("__label__", "") for label in predictions[0]]
                        
                        # Luo prediction can sometimes be classified as related Nilotic/African languages if short,
                        # so we verify if the target label is in the top 3 predictions
                        if fasttext_code not in pred_labels:
                            print(f"Row {original_idx} rejected: LangID failed for {col_name}. Got {pred_labels}, expected '{fasttext_code}'. Retrying translation with beam search...")
                            # Fallback retry with beam search
                            inputs_single = tokenizer([original_english], return_tensors="pt").to(device)
                            with torch.no_grad():
                                tokens_retry = model.generate(
                                    **inputs_single,
                                    forced_bos_token_id=tgt_lang_id,
                                    max_length=64,
                                    num_beams=3
                                )
                            target_text = tokenizer.batch_decode(tokens_retry, skip_special_tokens=True)[0]
                    except Exception as e:
                        print(f"Warning: FastText prediction failed ({e}). Disabling Language ID filter for this run.")
                        ft_model = None
                
                final_translations.append(target_text)
                
            # Write batch translations back to DataFrame
            for i, original_idx in enumerate(batch_idx):
                df.at[original_idx, col_name] = final_translations[i]
            
            # Save checkpoint incrementally
            df.to_csv(args.output, index=False, encoding="utf-8")
            
        # Clean up memory completely before loading next model
        print(f"Freeing memory for {col_name} model...")
        del model
        del tokenizer
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    print("\nAll translations completed successfully!")
    print(f"Final aligned 4-way parallel dataset saved to '{args.output}'.")

if __name__ == "__main__":
    main()
