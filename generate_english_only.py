import os
import argparse
from config import DEFAULT_SIZE, RANDOM_SEED
from utils import set_seed
from generator import PSAGenerator
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Generate English PSAs for Colab Translation")
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE, help="Total number of PSA pairs to generate")
    parser.add_argument("--output", type=str, default="output/english_psas.csv", help="Path to output English CSV file")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed for reproducibility")
    
    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Initialize generator (we pass None as translator since we won't translate locally)
    generator = PSAGenerator(size=args.size, translator=None)
    
    # Generate English PSAs
    print(f"Generating {args.size} unique validated English PSAs...")
    english_records = generator.generate_english_psas()
    
    # Save to CSV
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df = pd.DataFrame(english_records)
    df.to_csv(args.output, index=False, encoding="utf-8")
    print(f"Saved {len(df)} English records to {args.output}")
    print("\nNext steps:")
    print("1. Upload this file ('english_psas.csv') and 'translate_colab.py' to your Google Colab workspace.")
    print("2. Run the Colab notebook using GPU acceleration.")

if __name__ == "__main__":
    main()
