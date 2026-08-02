# Synthetic Parallel English-Swahili-Somali-Luo PSA Generator

An optimized, high-throughput pipeline designed to generate a dataset of over 50,000 high-quality, 4-way parallel Public Service Announcement (PSA) records in English, Swahili, Somali, and Luo for Machine Translation (MT) training.

---

## 📊 Datasets

The generated datasets are directly available in the repository:
*   [output/english_psas.csv](file:///C:/Users/Admin/.gemini/antigravity-ide/scratch/psa_generator/output/english_psas.csv): Contains the generated unique English announcements.
*   [output/psa_parallel_dataset.csv](file:///C:/Users/Admin/.gemini/antigravity-ide/scratch/psa_generator/output/psa_parallel_dataset.csv): The final 4-way parallel Swahili-Somali-Luo-English dataset ready for MT training.

---

## 🚀 How it Works

The pipeline is split into two specialized stages to maximize quality and performance:
1.  **Constrained LLM Generation**: English announcements are custom-written by Azure OpenAI GPT-4o. The prompt enforces strict negative constraints to match the active, punchy style of real-world Kenyan PSAs, while a Python **Balancing Controller** guarantees uniform data distribution across domains.
2.  **GPU Cloud Translation**: The English seed set is translated into Swahili, Somali, and Luo sequentially on a Google Colab GPU using Meta's **NLLB-200** models (distilled-600M for Swahili, and 1.3B for Somali and Luo). Key Kenyan acronyms (SHA, HELB, NTSA) are protected, and the translation runs in parallel batches for 100x faster execution.

---

## 🛠️ Project Structure

```
psa_generator/
│
├── generate_english_only.py # Script to generate raw English CSV via Azure LLM
├── translate_colab.py       # GPU-accelerated NLLB translation script (for Colab)
├── verify_dataset.py        # Validates columns, format, and null values in final CSV
├── test_generator.py        # Automated unit testing suite
├── requirements.txt         # Project python dependencies
├── pipeline.ipynb           # Complete Colab notebook pipeline runner
├── report.md                # Detailed methodology and synthesis report
│
├── src/                     # Core codebase package
│   ├── __init__.py          # Package initializer
│   ├── config.py            # Configuration settings (dataset size, paths, thresholds)
│   ├── generator.py         # Orchestrator for concurrent LLM generation
│   ├── llm_generator.py     # Azure OpenAI client and validation rewrite loop
│   ├── validator.py         # Enforces word lengths (10-25) and rejects passive/If clauses
│   ├── deduplicator.py      # Prevents exact and near-duplicate text generation
│   ├── exporter.py          # Randomly interleaves rows across domains to prevent blocking
│   ├── translator.py        # NLLB local/cloud translation wrapper
│   ├── utils.py             # Shared helpers (e.g., seeding for reproducibility)
│   │
│   ├── templates/           # Configuration templates for metadata balancing
│   │   ├── __init__.py
│   │   └── families.py      # Metadata template definitions
│   │
│   └── knowledge/           # Structured lists of entities, locations, and actions
│       ├── __init__.py
│       ├── scenarios.py     # Domain scenarios mapping
│       └── entities.py      # Entity objects
```

---

## 📖 Complete Setup & Execution Guide

### Part 1: Prerequisites & Keys
To generate the dataset, you will need:
*   An **Azure AI Foundry API Key** for GPT-4o.
*   The deployment endpoint and deployment name of your model.

### Part 2: Colab Notebook Execution (Recommended)
We have provided a pre-configured [pipeline.ipynb](file:///C:/Users/Admin/.gemini/antigravity-ide/scratch/psa_generator/pipeline.ipynb) notebook in the repository root to automate the generation and translation:

1.  Open **[Google Colab](https://colab.research.google.com/)**.
2.  Click **File** > **Upload Notebook** and upload the `pipeline.ipynb` file from your repository clone.
3.  Go to **Runtime** > **Change runtime type**, select **T4 GPU** (free tier), and click **Save**.
4.  Run the cells sequentially:
    *   **Step 0**: Clones/Pulls the latest code.
    *   **Step 1**: Installs dependencies.
    *   **Step 2**: Prompts for your Azure API key and runs `generate_english_only.py` to synthesize the English PSAs.
    *   **Step 3**: Sequentially translates Swahili, Somali, and Luo on the GPU using `translate_colab.py`.
    *   **Step 5**: Saves, commits, and pushes the final dataset directly back to your GitHub repository.

---

## 🧪 Verification & Tests

To run the unit tests locally:

```bash
python -m unittest test_generator.py
```
*(Runs core verification tests utilizing mocked LLM generators, completing successfully in ~0.2s).*
