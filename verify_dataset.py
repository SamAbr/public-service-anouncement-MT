import os
import pandas as pd

def verify_dataset(filepath="output/psa_parallel_dataset.csv"):
    if not os.path.exists(filepath):
        print(f"Error: Dataset file '{filepath}' not found. Please run the generation pipeline first.")
        return False
        
    print(f"Loading dataset from '{filepath}'...")
    df = pd.read_csv(filepath)
    
    # 1. Check columns
    expected_cols = ["PSA_Id", "Domain", "Class", "English", "Kiswahili", "Somali", "Luo", "is_synthetic", "model_version", "template_id"]
    missing_cols = [col for col in expected_cols if col not in df.columns]
    
    if missing_cols:
        print(f"[FAILED] Missing columns in CSV: {missing_cols}")
        return False
    print("[PASSED] All 10 expected columns exist.")
    
    # 2. Check size
    print(f"Total records found: {len(df)}")
    
    # 3. Check for empty cells
    failed = False
    for col in expected_cols:
        null_count = df[col].isna().sum()
        empty_count = (df[col] == "").sum()
        total_empty = null_count + empty_count
        if total_empty > 0:
            print(f"[FAILED] Column '{col}' contains {total_empty} empty/null values.")
            failed = True
            
    if not failed:
        print("[PASSED] Zero empty or null values found in any columns.")
        
    # 4. Check synthetic and model flags
    is_synthetic_val = df["is_synthetic"].unique()
    model_version_val = df["model_version"].unique()
    print(f"is_synthetic values: {is_synthetic_val}")
    print(f"model_version values: {model_version_val}")
    
    # 5. Display a few records for visual confirmation
    print("\nSample Parallel Row:")
    print("-" * 80)
    sample = df.iloc[0]
    for col in expected_cols:
        print(f"[{col}]: {sample[col]}")
    print("-" * 80)
    
    return not failed

if __name__ == "__main__":
    verify_dataset()
