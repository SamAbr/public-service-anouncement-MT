import argparse
import sys
import os
from src.config import DEFAULT_SIZE, OUTPUT_FILE, BATCH_SIZE, RANDOM_SEED, MODEL_NAME
from src.utils import set_seed
from src.generator import PSAGenerator
from src.exporter import Exporter
from src.translator import NLLBTranslator

def check_dependencies():
    """Checks if the required ML packages are installed."""
    missing = []
    try:
        import torch
    except ImportError:
        missing.append("torch")
    try:
        import transformers
    except ImportError:
        missing.append("transformers")
    try:
        import sentencepiece
    except ImportError:
        missing.append("sentencepiece")
    try:
        import pandas
    except ImportError:
        missing.append("pandas")
    try:
        import tqdm
    except ImportError:
        missing.append("tqdm")
        
    if missing:
        print("Error: The following required packages are missing:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\nPlease run the following command to install them:")
        print(f"  pip install -r {os.path.join(os.path.dirname(os.path.abspath(__file__)), 'requirements.txt')}")
        sys.exit(1)

def main():
    # Verify dependencies are present
    check_dependencies()

    parser = argparse.ArgumentParser(description="Synthetic Parallel English-Swahili PSA Generator")
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE, help="Total number of PSA pairs to generate")
    parser.add_argument("--output", type=str, default=OUTPUT_FILE, help="Path to output CSV file")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size for translation model")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed for reproducibility")
    
    args = parser.parse_args()

    print("=" * 60)
    print("      Synthetic Parallel English-Swahili PSA Generator")
    print("=" * 60)
    print(f"  Target size : {args.size}")
    print(f"  Output path : {args.output}")
    print(f"  Batch size  : {args.batch_size}")
    print(f"  Random seed : {args.seed}")
    print("=" * 60)

    # Set seed for reproducibility
    set_seed(args.seed)

    # Setup translator
    translator = NLLBTranslator(
        model_name=MODEL_NAME,
        batch_size=args.batch_size
    )

    # Initialize generator
    generator = PSAGenerator(
        size=args.size,
        translator=translator
    )

    # Run generation and translation
    records = generator.generate_and_translate()

    # Export records to CSV
    exporter = Exporter(output_file=args.output)
    exporter.export(records)

    print("\nGeneration pipeline finished successfully!")

if __name__ == "__main__":
    main()
