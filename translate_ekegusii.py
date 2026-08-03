import argparse
import os
import pandas as pd
from src.retriever import FewShotRetriever

def main():
    parser = argparse.ArgumentParser(description="Dynamic Retrieval-Augmented Few-Shot Translation to Ekegusii")
    parser.add_argument("--input", type=str, default="output/english_psas.csv", help="Input English CSV file")
    parser.add_argument("--sample-size", type=int, default=5, help="Number of samples to print instructions/prompts for")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        return

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} records from {args.input}.\n")

    print("Generating Dynamic Few-Shot Prompts for your first few PSAs:\n")
    print("=" * 80)
    for idx, row in df.head(args.sample_size).iterrows():
        english_psa = row["English"]
        retrieved_examples = FewShotRetriever.retrieve(english_psa, top_k=3)
        
        print(f"\nPSA #{idx+1} Input:")
        print(f"  {english_psa}\n")
        print("Retrieved Semantically Relevant Few-Shot Context:")
        for i, example in enumerate(retrieved_examples):
            print(f"  {i+1}. [{example['domain']}] EN: {example['English']}")
            print(f"     -> Ekegusii: {example['Ekegusii']}")
        print("-" * 80)

if __name__ == "__main__":
    main()
