# Synthetic Parallel English–Swahili PSA Generator

An optimized, modular pipeline designed to generate a dataset of over 50,000 high-quality parallel Public Service Announcement (PSA) pairs in English and Standard Kenyan Swahili for Machine Translation (MT) training.

---

## 📊 Datasets

The pre-generated datasets are directly available in the repository:
*   [output/english_psas.csv](file:///C:/Users/Admin/.gemini/antigravity-ide/scratch/psa_generator/output/english_psas.csv) (50,000 generated unique English announcements).
*   [output/psa_parallel_dataset.csv](file:///C:/Users/Admin/.gemini/antigravity-ide/scratch/psa_generator/output/psa_parallel_dataset.csv) (The final 50,000 parallel English-Swahili dataset ready for training).

---

## 🚀 How it Works

The pipeline splits generation and translation to maximize performance while remaining **100% free**:
1.  **Local Generation**: A controlled grammar and template-based engine generates 50,000 unique, validated English announcements locally in seconds.
2.  **GPU Cloud Translation**: The English announcements are uploaded to **Google Colab** and translated into Swahili using Meta's open-source **NLLB-200 (distilled-600M)** model. The translation runs in **FP16 precision**, completing all 50,000 translations in **under 10 minutes** on Colab's free T4 GPU.

---

## 🛠️ Project Structure

```
psa_generator/
│
├── main.py                  # Local generator CLI entry point
├── generate_english_only.py # Script to output raw English CSV for upload
├── translate_colab.py       # GPU-accelerated NLLB translation script (for Colab)
├── test_generator.py        # Automated unit testing suite
├── requirements.txt         # Project python dependencies
│
├── src/                     # Core codebase package
│   ├── __init__.py          # Package initializer
│   ├── config.py            # Configuration settings (dataset size, paths, thresholds)
│   ├── generator.py         # Core orchestrator for unique English generation
│   ├── grammar.py           # Controlled grammar engine (combining sentence slots)
│   ├── validator.py         # Enforces length (25-60 words) and formal styling rules
│   ├── deduplicator.py      # Prevents exact and near-duplicate text generation
│   ├── exporter.py          # Randomly interleaves rows across domains to prevent blocking
│   ├── translator.py        # NLLB local/cloud translation wrapper
│   ├── utils.py             # Shared helpers (e.g., seeding for reproducibility)
│   │
│   ├── templates/           # Domain-specific controlled sentence structures
│   │   ├── __init__.py
│   │   ├── education.py
│   │   ├── agriculture.py
│   │   ├── governance.py
│   │   ├── health.py
│   │   └── security.py
│   │
│   └── knowledge/           # Structured lists of entities, locations, and actions
│       ├── __init__.py
│       ├── domains.py       # Domain definitions mapping
│       ├── institutions.py  # Official Kenyan institutions (HELB, NTSA, KEPHIS, etc.)
│       ├── audiences.py     # Domain-specific target groups
│       ├── actions.py       # Standard directives and advice clauses
│       ├── hazards.py       # Context issues (phishing, flash floods, cholera, etc.)
│       ├── locations.py     # Kenyan geographical context locations
│       └── terminology.py   # Keyword vocabulary
└── output/
    ├── english_psas.csv     # Pre-generated raw English announcements
    └── psa_parallel_dataset.csv # Final Swahili-English parallel dataset
```

---

## 📖 Complete Setup & Execution Guide

### Part 1: Generate English PSAs Locally
Run these commands in your local computer's terminal:

1.  **Setup Virtual Environment**:
    ```bash
    python -m venv .venv
    # On Windows (native):
    .\.venv\Scripts\activate
    # On Unix/MSYS2:
    source .venv/bin/activate
    ```
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Generate English CSV**:
    ```bash
    python generate_english_only.py --size 50000 --output output/english_psas.csv
    ```
    *This generates `output/english_psas.csv` containing 50,000 unique and validated English PSAs in just 2-3 seconds.*

---

### Part 2: GPU-Accelerated Translation on Google Colab

1.  Open **[Google Colab](https://colab.research.google.com/)** and create a new notebook.
2.  Go to **Runtime** > **Change runtime type**, select **T4 GPU** (free tier), and click **Save**.
3.  Copy and run this single code block in Colab to upload your files, install libraries, and translate:

```python
# 1. Upload files programmatically from your computer
from google.colab import files
print("Please upload 'english_psas.csv' and 'translate_colab.py':")
uploaded = files.upload()

# 2. Install dependencies
!pip install transformers sentencepiece tqdm pandas torch

# 3. Run GPU translation in fast FP16 precision
!python translate_colab.py --input english_psas.csv --output psa_parallel_dataset.csv --batch-size 128
```

4.  When prompted, select the locally generated `output/english_psas.csv` and the `translate_colab.py` script.
5.  After the translation completes (~5-10 minutes), download your parallel dataset:
```python
from google.colab import files
files.download('psa_parallel_dataset.csv')
```

---

## 🧪 Verification & Tests

To run the unit tests and verify the generator's template parsing, deduplication, and validation rules locally:

```bash
python -m unittest test_generator.py
```
*(Runs 4 core verification tests utilizing a mock translator, completing successfully in ~0.02s).*
