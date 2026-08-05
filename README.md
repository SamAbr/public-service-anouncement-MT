# Fine-Tuning Neural Machine Translation Models for Kenyan Public Service Announcements Using a Synthetic Parallel Corpus

> **United States International University–Africa** · Department of Computing · 2026

---

## 👥 Team

| Name | Role |
|------|------|
| Weldesenbet Zeray | Team Member |
| Samuel Abrha | Team Member |
| Hetal Kumbharana | Team Member |
| Halima Mohammed | Team Member |
| Peter Kidiga | Team Member |
| Mitchelle Moraa | Team Member |

**Supervisor:** Professor Edward Ombui

---

## 📌 Overview

This project addresses the challenge of domain-specific machine translation for **low-resource Kenyan languages**. General-purpose MT systems struggle with the specialised style of Public Service Announcements (PSAs) because they are trained on broad multilingual data rather than domain-specific content.

We built an end-to-end pipeline that:

1. **Synthesises** a large-scale English PSA corpus using a constrained LLM (GPT-5 mini via Azure OpenAI).
2. **Translates** the corpus into Kiswahili and Somali using NLLB-200.
3. **Fine-tunes** NLLB-200 and mBART-50 on the generated parallel corpus.
4. **Evaluates** translation quality using BLEU and chrF metrics.

---

## 📊 Datasets

All generated datasets are available in the `output/` directory and committed to the repository:

| File | Description |
|------|-------------|
| `output/english_psas.csv` | 50,318 validated English PSAs |
| `output/psa_parallel_dataset.csv` | Final parallel corpus (English · Kiswahili · Somali · Luo) |
| `output/scraped_english_ekegusii.csv` | Extracted English-Ekegusii verse pairs |

---

## 🏆 Results

### Fine-Tuned vs. Zero-Shot Translation Performance

| Model | BLEU | chrF |
|-------|------|------|
| NLLB-200 Zero-shot | 30.83 | 60.42 |
| **NLLB-200 Fine-tuned** ⭐ | **50.86** | **73.37** |
| mBART-50 Zero-shot | 0.18 | 12.86 |
| mBART-50 Fine-tuned | 9.14 | 28.22 |

**Key Findings:**
- 🏆 **NLLB-200 fine-tuned** achieved the best overall performance: BLEU 50.86 / chrF 73.37.
- 📈 **+64.7% BLEU improvement** over the NLLB-200 zero-shot baseline after fine-tuning.
- mBART-50 showed improvement but significantly underperformed NLLB-200 in the PSA domain.

---

## 🚀 Pipeline Architecture

### Phase 1: English Corpus Generation

```
Scenario Knowledge Base
         │
         ▼
   Metadata Planner  ──► Domain-balanced permutation (Health, Education, Agriculture, Governance, Safety)
         │
         ▼
 GPT-5 mini (Azure OpenAI)  ──► Constrained few-shot synthesis (10 parallel workers)
         │
         ▼
   Validation Pipeline
   ├── Single sentence, 10–25 words
   ├── One clear public action
   ├── Public advisory style
   ├── No Gazette / press-release language
   └── Jaccard deduplication
         │
         ▼
 50,318 Validated English PSAs
```

### Phase 2: NMT Translation (NLLB-200, GPU)

```
English PSAs
      │
      ▼
 NLLB-200 (Google Colab GPU)
      │
      ├──────► Kiswahili  (nllb-200-distilled-600M)
      ├──────► Somali     (nllb-200-1.3B)
      └──────► Luo        (nllb-200-1.3B)
```

- **Acronym Protection**: Custom tokenizer tokens registered for `SHA`, `HELB`, `NTSA`, `KRA`, `Huduma`.
- **Google Drive Resumability**: Output CSV checkpointed to Drive — safely resumes after runtime disconnect.

### Phase 3: Model Fine-Tuning

```
Parallel Corpus  →  Tokenization  →  Mini-batch Training

     →  Forward Pass  →  Loss Computation
     →  Backpropagation  →  Parameter Update
     →  Fine-Tuned Model
```

Models fine-tuned:
- `facebook/nllb-200-distilled-600M` → **NLLB-200 Fine-tuned**
- `facebook/mbart-large-50-many-to-many-mmt` → **mBART-50 Fine-tuned**

---

## 🛠️ Project Structure

```
psa_generator/
│
├── english/
│   └── generate_english_only.py    # English PSA synthesis via Azure LLM
│
├── corpus_translation/
│   └── translate_colab.py          # GPU-accelerated NLLB translation (Swahili, Somali, Luo)
│
├── ekegusii/
│   └── scrape_ekegusii_corpus.py   # Scrapes Ekegusii parallel corpus from web sources
│
├── tests/
│   ├── verify_dataset.py           # Validates columns, format, and null values
│   └── test_generator.py           # Automated unit test suite
│
├── requirements.txt                # Python dependencies
├── pipeline.ipynb                  # Colab notebook: English generation + Swahili/Somali/Luo NMT
├── report.md                       # Detailed methodology report
│
├── src/
│   ├── config.py                   # Configuration settings
│   ├── generator.py                # Concurrent LLM generation orchestrator
│   ├── llm_generator.py            # Azure OpenAI client + validation rewrite loop
│   ├── validator.py                # Word length + style constraint enforcement
│   ├── deduplicator.py             # Jaccard-based near-duplicate filtering
│   ├── exporter.py                 # Domain-interleaved row export
│   ├── translator.py               # NLLB translation wrapper
│   ├── templates/families.py       # Metadata template definitions
│   └── knowledge/
│       ├── scenarios.py            # Domain scenario mappings
│       ├── entities.py             # Kenyan entity objects
│       └── ekegusii_fewshot_corpus.py  # Hand-curated Ekegusii seed pairs
│
└── output/
    ├── english_psas.csv
    ├── psa_parallel_dataset.csv
    └── scraped_english_ekegusii.csv
```

---

## 📖 Setup & Execution Guide

### Prerequisites

- An **Azure AI Foundry API Key** (GPT-5 mini deployment).
- A Google account (for Google Colab GPU).

### Setup: Swahili / Somali / Luo Pipeline (GPU)

1. Open [Google Colab](https://colab.research.google.com/) and upload **`pipeline.ipynb`**.
2. Set runtime to **T4 GPU** → *Runtime > Change runtime type*.
3. Run cells sequentially:
   - **Step 0**: Clone/pull the repository.
   - **Step 1**: Install dependencies.
   - **Step 2**: Generate 50,318 English PSAs (paste your Azure API key).
   - **Step 3**: Mount Google Drive for resumable checkpoint storage.
   - **Step 4**: Translate to Kiswahili, Somali, and Luo via NLLB-200.
   - **Step 5**: Download or push the dataset to GitHub.

---

## 🧪 Tests

```bash
python -m unittest test_generator.py
```

---

## 📚 Key Technologies

| Technology | Role |
|------------|------|
| **GPT-5 mini (Azure OpenAI)** | English PSA synthesis & Ekegusii translation |
| **NLLB-200** | Zero-shot & fine-tuned NMT (Swahili, Somali, Luo) |
| **mBART-50** | Comparative fine-tuning baseline |
| **BLEU** | N-gram precision evaluation metric |
| **chrF** | Character-level F-score evaluation metric |
| **Google Colab (T4 GPU)** | GPU-accelerated NMT translation |

---

## 🔭 Future Work

- Expand the parallel corpus to additional Kenyan languages (Kikuyu, Kamba, Kalenjin).
- Conduct human evaluation with native speakers.
- Fine-tune larger multilingual models on the expanded corpus.
- Deploy fine-tuned models as a web-based translation API for Kenyan county governments.

---

## 📄 License

This project is for academic research purposes at **USIU–Africa**. Dataset and code are published to support future research in multilingual MT for under-resourced African languages.
