import os
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer, 
    DataCollatorForSeq2Seq
)

def train_machine_translation(
    dataset_path="output/psa_parallel_dataset.csv",
    model_name="facebook/nllb-200-distilled-600M",
    output_dir="output/fine_tuned_nllb",
    target_lang="swh_Latn"  # e.g., swh_Latn (Swahili), som_Latn (Somali), luo_Latn (Luo)
):
    """
    Template training script to fine-tune NLLB-200 on the generated PSA parallel dataset.
    """
    if not os.path.exists(dataset_path):
        print(f"Error: Parallel dataset '{dataset_path}' not found. Please run the generation and translation steps first!")
        return

    print(f"=== Loading dataset from '{dataset_path}' ===")
    df = pd.read_csv(dataset_path)
    
    # Map target columns
    src_col = "English"
    if target_lang == "swh_Latn":
        tgt_col = "Kiswahili"
    elif target_lang == "som_Latn":
        tgt_col = "Somali"
    elif target_lang == "luo_Latn":
        tgt_col = "Luo"
    else:
        raise ValueError(f"Unsupported target language: {target_lang}")

    # Remove any empty/null rows for the target translation
    df = df.dropna(subset=[src_col, tgt_col])
    print(f"Total training pairs available: {len(df)}")

    # Split train/eval
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    print(f"Split: Train={len(train_df)} | Eval={len(val_df)}")

    # 1. Load Tokenizer & Model
    print(f"Loading pre-trained model and tokenizer: '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="eng_Latn", tgt_lang=target_lang)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # 2. Tokenization helper
    def preprocess_function(examples):
        inputs = [ex for ex in examples[src_col]]
        targets = [ex for ex in examples[tgt_col]]
        model_inputs = tokenizer(inputs, max_length=128, truncation=True)

        # Set up the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=128, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # Convert pandas DataFrames to tokenized Hugging Face dataset format
    from datasets import Dataset
    train_dataset = Dataset.from_pandas(train_df[[src_col, tgt_col]])
    val_dataset = Dataset.from_pandas(val_df[[src_col, tgt_col]])

    train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=[src_col, tgt_col])
    val_dataset = val_dataset.map(preprocess_function, batched=True, remove_columns=[src_col, tgt_col])

    # 3. Data Collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # 4. Define Seq2Seq Training Arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=3,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),  # Enable FP16 training if running on GPU
        push_to_hub=False,
        report_to="none"
    )

    # 5. Initialize Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # 6. Start Fine-Tuning
    print("\nStarting Fine-Tuning loop...")
    # To run actual training in Colab, uncomment the line below:
    # trainer.train()
    # trainer.save_model(os.path.join(output_dir, "final_model"))
    print(f"Model training configuration complete. Output directory set to: '{output_dir}'")

if __name__ == "__main__":
    train_machine_translation()
