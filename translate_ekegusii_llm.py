import argparse
import os
import time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI
from src.retriever import FewShotRetriever

def translate_single_sentence(client, model, english_text, top_k=3):
    # Dynamically retrieve the top semantic examples
    examples = FewShotRetriever.retrieve(english_text, top_k=top_k)
    
    # Construct few-shot prompt
    few_shot_block = ""
    for i, ex in enumerate(examples):
        few_shot_block += f"Example {i+1}:\n"
        few_shot_block += f"English: {ex['English']}\n"
        few_shot_block += f"Ekegusii: {ex['Ekegusii']}\n\n"
        
    system_prompt = f"""You are an expert English-to-Ekegusii (Kisii) translator. 
Translate the input English public service announcement (PSA) into natural Ekegusii.
Keep the translation action-oriented, clear, and command-focused. 

Here are verified reference translations similar to your input:
{few_shot_block}
Translate the following English sentence. Output ONLY the translation and nothing else. No introductions, no explanations, no quotes."""

    # Call LLM
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Translate this: {english_text}"}
            ],
            temperature=0.1,
            max_tokens=100
        )
        return response.choices[0].message.content.strip().strip('"')
    except Exception as e:
        print(f"Error translating sentence '{english_text}': {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Parallel Dynamic Retrieval-Augmented Ekegusii LLM Translation")
    parser.add_argument("--input", type=str, default="output/english_psas.csv", help="Input English CSV file")
    parser.add_argument("--output", type=str, default="output/psa_parallel_dataset.csv", help="Output parallel CSV file")
    parser.add_argument("--api-key", type=str, default=None, help="OpenAI / Azure API Key")
    parser.add_argument("--endpoint", type=str, default=None, help="API Endpoint URL (if using custom/Azure endpoint)")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Model name to use (e.g. gpt-4o-mini)")
    parser.add_argument("--workers", type=int, default=10, help="Number of concurrent translation workers")
    parser.add_argument("--batch-save", type=int, default=100, help="Save to CSV every N translated records")
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY")
    if not api_key:
        print("Error: API Key is required. Please provide it via --api-key or set the OPENAI_API_KEY environment variable.")
        return

    # Initialize client
    if args.endpoint:
        client = OpenAI(api_key=api_key, base_url=args.endpoint)
    else:
        client = OpenAI(api_key=api_key)

    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        return

    # Read/Initialize parallel dataset
    df_input = pd.read_csv(args.input)
    
    if os.path.exists(args.output):
        df = pd.read_csv(args.output)
        # Verify alignment
        if "English" not in df.columns or len(df) != len(df_input) or not df["English"].equals(df_input["English"]):
            print("Existing output parallel dataset does not match input English file. Initializing a fresh copy.")
            df = df_input.copy()
    else:
        df = df_input.copy()

    # Ensure Ekegusii column exists
    if "Ekegusii" not in df.columns:
        df["Ekegusii"] = ""

    # Identify untranslated indices
    untranslated_indices = df[df["Ekegusii"].isna() | (df["Ekegusii"] == "")].index.tolist()
    total_to_translate = len(untranslated_indices)
    
    if total_to_translate == 0:
        print("Ekegusii is already fully translated in the output dataset!")
        return

    print(f"Starting parallel translation to Ekegusii using model {args.model}.")
    print(f"Remaining records to translate: {total_to_translate} / {len(df)}")
    
    # Run parallel translations using ThreadPoolExecutor
    completed_count = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_idx = {
            executor.submit(translate_single_sentence, client, args.model, df.at[idx, "English"]): idx
            for idx in untranslated_indices
        }
        
        for future in tqdm(as_completed(future_to_idx), total=total_to_translate, desc="Translating"):
            idx = future_to_idx[future]
            try:
                translation = future.result()
                if translation:
                    df.at[idx, "Ekegusii"] = translation
            except Exception as e:
                print(f"Worker exception for row {idx}: {e}")
                
            completed_count += 1
            if completed_count % args.batch_save == 0:
                df.to_csv(args.output, index=False, encoding="utf-8")

    # Final save
    df.to_csv(args.output, index=False, encoding="utf-8")
    print(f"\nAll translations completed successfully! Saved to '{args.output}'.")

if __name__ == "__main__":
    main()
