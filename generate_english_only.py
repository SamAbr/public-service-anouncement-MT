import os
import argparse
import pandas as pd
from src.config import DEFAULT_SIZE, RANDOM_SEED
from src.utils import set_seed
from src.generator import PSAGenerator

def main():
    parser = argparse.ArgumentParser(description="Generate English PSAs for Colab Translation")
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE, help="Total number of PSA records to generate")
    parser.add_argument("--output", type=str, default="output/english_psas.csv", help="Path to output English CSV file")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed for reproducibility")
    parser.add_argument("--engine", type=str, default="templates", choices=["templates", "azure_llm"], help="Generation engine: templates or azure_llm")
    
    # Azure OpenAI credentials
    parser.add_argument("--azure-key", type=str, default=None, help="Azure OpenAI API Key")
    parser.add_argument("--azure-endpoint", type=str, default=None, help="Azure OpenAI Endpoint URL")
    parser.add_argument("--azure-deployment", type=str, default=None, help="Azure OpenAI GPT-4o Deployment Name")
    
    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Initialize generator with selected engine and credentials
    generator = PSAGenerator(
        size=args.size,
        translator=None,
        engine=args.engine,
        azure_api_key=args.azure_key,
        azure_endpoint=args.azure_endpoint,
        azure_deployment=args.azure_deployment
    )
    
    # Generate English PSAs
    print(f"Generating {args.size} unique validated English PSAs using engine: '{args.engine}'...")
    english_records = generator.generate_english_psas()
    
    # Save to CSV
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df = pd.DataFrame(english_records)
    df.to_csv(args.output, index=False, encoding="utf-8")
    print(f"Saved {len(df)} English records to {args.output}")

if __name__ == "__main__":
    main()
