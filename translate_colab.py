import argparse
import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm
import math
import os

def main():
    parser = argparse.ArgumentParser(description="Colab GPU Translation for NLLB-200")
    parser.add_argument("--input", type=str, default="english_psas.csv", help="Path to input English CSV file")
    parser.add_argument("--output", type=str, default="psa_parallel_dataset.csv", help="Path to output parallel CSV file")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for GPU translation")
    parser.add_argument("--model-name", type=str, default="facebook/nllb-200-distilled-600M", help="NLLB model name")
    
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        return

    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running translation on device: '{device}'")
    if device == "cpu":
        print("Warning: GPU not detected. Translation on Colab CPU will be slow. Switch your runtime to T4 GPU!")

    # Read English dataset
    print(f"Reading English records from '{args.input}'...")
    df = pd.read_csv(args.input)
    english_texts = df["English"].tolist()
    total_records = len(english_texts)
    print(f"Loaded {total_records} records.")

    # Load model and tokenizer
    print(f"Loading model '{args.model_name}' in FP16 precision...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, src_lang="eng_Latn")
    # Using torch.float16 speeds up T4 GPU inference by ~8x compared to default FP32
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name, 
        torch_dtype=torch.float16
    ).to(device)
    print("Model loaded successfully.")

    # Translate
    translated_texts = []
    num_batches = math.ceil(total_records / args.batch_size)
    tgt_lang_id = tokenizer.convert_tokens_to_ids("swh_Latn")

    print(f"Starting translation of {total_records} sentences in {num_batches} batches...")
    
    for i in tqdm(range(num_batches), desc="Translating"):
        batch_texts = english_texts[i * args.batch_size : (i + 1) * args.batch_size]
        
        # Tokenize
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(device)
        
        # Generate translation
        with torch.no_grad():
            translated_tokens = model.generate(
                **inputs,
                forced_bos_token_id=tgt_lang_id,
                max_length=64, # Optimized max token limit for PSAs (reduces redundant padding computation)
                num_beams=1  # Greedy decoding for maximum speed
            )
            
        # Decode
        batch_translations = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        translated_texts.extend(batch_translations)

    # Add translated Kiswahili column
    df["Kiswahili"] = translated_texts

    # Save final dataset
    print(f"Saving final parallel dataset to '{args.output}'...")
    df.to_csv(args.output, index=False, encoding="utf-8")
    print("Done! You can now download the translated parallel dataset.")

if __name__ == "__main__":
    main()
