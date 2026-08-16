import os
import argparse
import pandas as pd
from src.generator import PSAGenerator

DEFAULT_SIZE = 100
RANDOM_SEED = 42

def set_seed(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)

def main():
    parser = argparse.ArgumentParser(description="Generate English PSAs for Colab Translation")
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE, help="Total number of PSA records to generate")
    parser.add_argument("--output", type=str, default="output/english_psas.csv", help="Path to output English CSV file")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed for reproducibility")
    
    # Azure OpenAI credentials
    parser.add_argument("--azure-key", type=str, default=None, help="Azure OpenAI API Key")
    parser.add_argument("--azure-endpoint", type=str, default=None, help="Azure OpenAI Endpoint URL")
    parser.add_argument("--azure-deployment", type=str, default=None, help="Azure OpenAI GPT-4o Deployment Name")
    
    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Enforce Azure OpenAI credentials
    azure_key = args.azure_key or os.getenv("AZURE_OPENAI_API_KEY")
    if not azure_key:
        raise ValueError(
            "Azure API Key is required for generation. "
            "Please provide it via --azure-key or set the AZURE_OPENAI_API_KEY environment variable."
        )

    # Initialize generator with selected engine (forced to azure_llm) and credentials
    generator = PSAGenerator(
        size=args.size,
        engine="azure_llm",
        azure_api_key=azure_key,
        azure_endpoint=args.azure_endpoint,
        azure_deployment=args.azure_deployment
    )
    
    # Load existing CSV to calculate start counts and prepare for appending
    start_counts = {}
    df_existing = pd.DataFrame()
    if os.path.exists(args.output):
        try:
            df_existing = pd.read_csv(args.output)
            if not df_existing.empty and "Domain" in df_existing.columns:
                start_counts = df_existing.groupby("Domain").size().to_dict()
                print(f"Detected existing records in '{args.output}'. Start offsets: {start_counts}")
                # Update generator start counts
                generator.start_counts = start_counts
        except Exception as e:
            print(f"Warning: Failed to load existing CSV: {e}. Starting fresh.")
    
    # Generate English PSAs
    print(f"Generating {args.size} unique validated English PSAs using Azure LLM engine (ChatGPT)...")
    english_records = generator.generate_english_psas()
    
    # Merge and save to CSV
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df_new = pd.DataFrame(english_records)
    
    if not df_existing.empty:
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new
        
    df_combined.to_csv(args.output, index=False, encoding="utf-8")
    print(f"Successfully appended {len(df_new)} records. Total records in '{args.output}': {len(df_combined)}")

if __name__ == "__main__":
    main()
