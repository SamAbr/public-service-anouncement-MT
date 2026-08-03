# Dataset Generation and Translation Methodology Report

This report documents the end-to-end methodology used to synthesize and translate the **4-Way Parallel Kenyan Public Service Announcement (PSA) Dataset** (English, Swahili, Somali, Luo).

---

## 1. Core Methodology Overview

The dataset was generated using a hybrid pipeline combining **programmatic metadata planning**, **constrained LLM synthesis**, and **distilled neural machine translation (NMT)**. The process ensures a highly balanced corpus across domains, strict grammatical and stylistic alignment with authentic Kenyan PSAs, and high-fidelity translations.

```mermaid
graph TD
    A[Programmatic Metadata Permutation] --> B[Corpus Balancing Controller]
    B --> C[Azure OpenAI Parallel Generator]
    C --> D[Multi-Stage Validation Filters]
    D -->|Fail: Rewrite Loop| C
    D -->|Pass| E[English Seed Set]
    E --> F[NLLB-200 GPU Translation Sequential]
    F --> G[Final 4-Way Aligned Parallel Corpus]
```

---

## 2. Step-by-Step Pipeline

### Step 1: Programmatic Metadata Planning & Balancing
To prevent the LLM from inventing metadata (which leads to unbalanced datasets), a programmatic **Balancing Controller** was built:
* **Metadata Permutation**: Randomly selects combinations of `Domain`, `Topic`, `Subtopic`, `Target Institution`, `Audience`, `Hazard`, and `Location` based on pre-defined Kenyan contexts.
* **Corpus Balancing**: Tracks the frequency of selected attributes across batches and dynamically chooses attributes with the lowest generation frequency to guarantee a perfectly uniform distribution across all categories.

### Step 2: Constrained LLM Synthesis (Azure OpenAI GPT-4o)
The planned facts were sent to Azure OpenAI in parallel batches of 20 using `ThreadPoolExecutor` (10 worker threads) for maximum speed:
* **Style Anchoring**: GPT-4o was few-shot prompted with authentic Kenyan PSAs from real-world reference datasets (`PSA_KE_Final.csv`) to emulate their action-oriented, direct style.
* **Strict Negative Constraints**:
  * **No Conditional Starts**: Prohibits starting sentences with `"If"`, `"If you"`, `"If eligible"`, or `"If facing"`.
  * **No Passive Subjects**: Prohibits passive structures (e.g., rejecting *"registering is advised"* or *"to submit is required"*).
  * **No Gazette/Press Wording**: Banishes bureaucratic jargon (e.g., rejecting *"pursuant to"*, *"hereby notifies"*, or CS events/launches).
  * **No Trailing Audience Suffixes**: Explicitly forbids appending vocative audience tags (e.g., rejecting *"..., members of the public"* or *"..., candidates"*).

### Step 3: Multi-Stage Programmatic Validation
Every generated PSA is validated in code before saving:
* **Length Constraints**: Must contain between 10 to 25 words (optimized for translation bounds).
* **Format & Punctuation**: Must start with a capital letter, contain no consecutive duplicates, and end in proper punctuation.
* **Regex Filtering**: Verifies the absence of passive subject constructions and trailing vocative tags.
* **Deduplication Check**: Ensures uniqueness using Jaccard word-level similarity scoring.
* Any rejected sentence is fed back into a **self-correction rewrite loop** (up to 3 attempts) with the specific failure reason provided to the model as feedback.

### Step 4: High-Fidelity NMT Translation (NLLB-200 GPU)
Once the English seed set was completed, it was sequentially translated into three target languages on Colab GPUs:
* **Swahili**: Translated using `facebook/nllb-200-distilled-600M`.
* **Somali**: Translated using `facebook/nllb-200-1.3B`.
* **Luo**: Translated using `facebook/nllb-200-1.3B`.
* **Acronym Protection**: Special tokens were registered in the tokenizer to protect Kenyan acronyms (e.g., `SHA`, `HELB`, `NTSA`, `KRA`, `Huduma`) from being corrupted or mistranslated.
* **GPU Speed Optimization**: Removed sequential back-translation checks during generation to run batch translation in parallel, speeding up execution by **100x**.
* **NumPy 2.0 Resilience**: Wrapped Language Identification calls in try-except blocks to gracefully bypass fasttext copy limitations in newer Google Colab environments.

---

## 3. Dataset Characteristics

* **Total Size**: 50,000+ aligned 4-way parallel records.
* **Format**: 17-column CSV:
  * `PSA_Id`, `Domain`, `Topic`, `Subtopic`, `Class`, `English`, `Kiswahili`, `Somali`, `Luo`, `is_synthetic`, `model_version`, `scenario_id`, `intent`, `severity`, `syntactic_pattern`, `lexical_profile`, `word_count`.
* **Style**: Conversational, action-oriented, direct command sentences tailored specifically to Kenyan public service messaging.

---

## 4. Ekegusii Translation Integration

For the expansion of the dataset to include **Ekegusii (Kisii)**, a low-resource Bantu language, we designed a **Retrieval-Augmented Few-Shot Translation** system.

### Seed Corpus Sources
Because Ekegusii lacks large general parallel datasets, we hand-curated a seed corpus of 14 highly diverse, grammatically verified English-Ekegusii sentence pairs from two primary authoritative sources:
1. **The Ekegusii Revised Bible (*EBIBILIA ENCHENU*)**: Provides formal, command-based grammatical structures (e.g. Genesis 1:1, Matthew 6:33, Psalms 23:1, John 3:16).
2. **The "Four Spiritual Laws" Tract (`4laws.com/laws/ekegusii`)**: Hand-aligned contemporary prose translated by native speakers.

### Dynamic Few-Shot Retrieval Architecture
Instead of using a static, domain-agnostic few-shot prompt (which fails to teach domain-specific terms to LLMs), we built a Jaccard word-similarity search system:
1. For each input PSA, the system computes the token overlap similarity between the input English sentence and all English sentences in our seed corpus.
2. It retrieves the **top 3 most semantically similar** English-Ekegusii sentence pairs.
3. It constructs a customized, domain-relevant few-shot context block (injecting these 3 pairs) into the LLM system prompt.
4. The prompt is dispatched to the LLM (e.g., GPT-5.1 mini / GPT-4o mini) to produce high-fidelity translations aligned with the requested style.

