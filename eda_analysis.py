import os
import pandas as pd
import matplotlib.pyplot as plt

def run_eda(filepath="output/psa_parallel_dataset.csv"):
    """
    Performs Exploratory Data Analysis (EDA) on the generated PSA Parallel Dataset.
    Logs key statistics and saves visualization plots.
    """
    if not os.path.exists(filepath):
        # Fallback to English dataset if parallel dataset is not yet translated
        filepath = "output/english_psas.csv"
        
    if not os.path.exists(filepath):
        print(f"Error: Dataset file '{filepath}' not found. Please run the generator first!")
        return

    print(f"=== Running EDA on '{filepath}' ===")
    df = pd.read_csv(filepath)
    
    print(f"\n1. Basic Dataset Information:")
    print(f"Total Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    print("\n2. Null Value Summary:")
    print(df.isnull().sum())
    
    # Analyze word counts
    if "word_count" in df.columns:
        print("\n3. Word Count Statistics:")
        print(df["word_count"].describe())
        
    # Distribution of categorical metadata
    for col in ["Domain", "intent", "severity", "syntactic_pattern"]:
        if col in df.columns:
            print(f"\n4. Distribution of '{col}':")
            dist = df[col].value_counts()
            print(dist)
            
            # Save distribution plots
            plt.figure(figsize=(8, 4))
            dist.plot(kind="bar", color="skyblue", edgecolor="black")
            plt.title(f"Distribution of PSA {col}")
            plt.ylabel("Count")
            plt.xlabel(col)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            os.makedirs("output/eda_plots", exist_ok=True)
            plot_path = f"output/eda_plots/{col.lower()}_distribution.png"
            plt.savefig(plot_path)
            print(f"Saved plot to '{plot_path}'")
            plt.close()

    print("\n=== EDA Analysis Completed Successfully! ===")

if __name__ == "__main__":
    run_eda()
