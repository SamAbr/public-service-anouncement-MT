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

### Step 2: Constrained LLM Synthesis (Azure OpenAI GPT-5.1 mini)
The planned facts were sent to Azure OpenAI in parallel batches of 20 using `ThreadPoolExecutor` (10 worker threads) for maximum speed:
* **Style Anchoring**: GPT-5.1 mini was few-shot prompted with authentic Kenyan PSAs from real-world reference datasets (`PSA_KE_Final.csv`) to emulate their action-oriented, direct style.
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
* **NumPy 2.0 Resilience**: Wrapped Language Identification calls in try-except blocks to gracefully bypass fasttext copy limitations in newer Google Colab environments.

---

## 3. Dataset Characteristics

* **Total Size**: 50,000+ aligned 4-way parallel records.
* **Format**: 17-column CSV:
  * `PSA_Id`, `Domain`, `Topic`, `Subtopic`, `Class`, `English`, `Kiswahili`, `Somali`, `Luo`, `is_synthetic`, `model_version`, `scenario_id`, `intent`, `severity`, `syntactic_pattern`, `lexical_profile`, `word_count`.
* **Style**: Conversational, action-oriented, direct command sentences tailored specifically to Kenyan public service announcements.

---

## 4. Ekegusii Translation Integration

For the expansion of the dataset to include **Ekegusii (Kisii)**, a low-resource Bantu language, we designed a **Retrieval-Augmented Few-Shot Translation** system.

### Seed Corpus Sources
Because Ekegusii lacks large general parallel datasets, we hand-curated a seed corpus of 14 highly diverse, grammatically verified English-Ekegusii sentence pairs from the **Ekegusii Revised Bible (*EBIBILIA ENCHENU*)**, which provides formal, command-based grammatical structures (e.g. Genesis 1:1, Matthew 6:33, Psalms 23:1, John 3:16).

The "Four Spiritual Laws" tract (`4laws.com/laws/ekegusii`) was evaluated as a second source but contributed **zero usable pairs**: its English and Ekegusii pages are aligned only by document position, not by sentence, and no extracted pair survived cleaning. It has been removed from the pipeline.

### Domain-Specific Static Few-Shot Prompting
Instead of dynamically retrieving examples per sentence (which is time and token consuming), we pre-configured a static system prompt for each of the core domains (Health, Agriculture/Environment, Security & Safety, Education, Governance) containing domain-relevant verified few-shot translations.

The translation script:
1. Iterates domain-by-domain.
2. Selectes the pre-configured prompt for the active domain.
3. Translates target records in parallel using ThreadPoolExecutor.

### Target Subset
For resource and cost optimization, only a subset of the English dataset is translated to Ekegusii:
* **First 15,000 records** (indices 0 to 14,999)
* **Last 15,000 records** (indices `len(df) - 15000` to `len(df) - 1`)
Any record outside this target range is left blank (`""`) for the Ekegusii column.


### Pipeline Separation
To prevent environment conflicts and provide a clean, user-friendly execution flow, the Colab notebooks are separated:
* **`pipeline.ipynb`**: Dedicated to GPU-accelerated sequential NMT translation for Swahili, Somali, and Luo.
* **`pipeline_ekegusii.ipynb`**: Dedicated to OpenAI/Azure LLM-driven Retrieval-Augmented Ekegusii translation.

---

## 5. Trilingual Bible Corpus (English · Ekegusii · Kiswahili)

To support fine-tuning NLLB-200 for Ekegusii — a language absent from NLLB-200's 202 supported languages — we built a three-way parallel corpus by verse-aligning USFM Bible editions from eBible.org. This is produced by `ekegusii/build_trilingual_corpus.py` and supersedes the earlier `scrape_ekegusii_corpus.py`.

### Source Selection
| Language | eBible ID | Edition | Rationale |
|----------|-----------|---------|-----------|
| English | `engbsb` | Berean Standard Bible | Public domain, contemporary register. Replaces the KJV, whose archaic English transfers poorly to plain-language PSA text. |
| Ekegusii | `guz` | Ekegusii Revised Bible 2021 | Only available Ekegusii scripture edition. |
| Kiswahili | `swhonen` | Neno: Bibilia Takatifu | Biblica open licence, modern Kiswahili. Preferred over the archaic 1850 union version. |

### Alignment and Filtering
Verses are keyed as `BOOK_CHAPTER_VERSE` and intersected across all three editions. Filtering applied in order:

* **USFM cleaning**: footnotes (`\f…\f*`), cross-references (`\x…\x*`) and figures are deleted as whole spans rather than untagged, preventing editorial apparatus from being merged into verse text.
* **Verse-span matching**: a verse is retained only when all three editions cover the identical span, so a merged `\v 1-2` on one side is never aligned against a bare `\v 1` on another.
* **Length constraints**: minimum 3 words (English) / 2 words (targets).
* **Length-ratio bounds**: cross-language character-length ratios must fall within calibrated bounds (generous upper limits to accommodate Bantu agglutination), catching residual misalignment.
* **Deduplication**: exact duplicate triples removed.

### Yield
| Stage | Count |
|-------|-------|
| Verses parsed (English / Ekegusii / Kiswahili) | 31,086 / 35,854 / 31,103 |
| Verse keys present in all three | 31,073 |
| Dropped — verse-span mismatch | 11 |
| Dropped — length-ratio outlier | 187 |
| Dropped — too short | 15 |
| Dropped — duplicate | 107 |
| **Bible Aligned Triples** | **30,753** |
| **Merged Storybook Triples** | **110** |
| **Total Combined Aligned Triples** | **30,863** |

### Supplementary Source: African Storybook Project
The African Storybook Project (`global-asp/asp-source`) holds 19 Ekegusii titles, of which 13 also exist in English and Kiswahili and 11 have matching page counts, yielding **121 page-aligned triples**. After manual filtering and discarding 11 page-level boilerplate/copyright lines, **110 clean triples** were successfully merged directly into `output/bible_en_guz_swh.csv` to enrich the training data with a contemporary register closer to public announcements.

All source IDs, stage counts, and per-filter drop counts are recorded in `output/corpus_manifest.json` for reproducibility.

---

## 6. Lughayangu Dictionary Scraper (ekegusii/scrape_lughayangu.py)

To supplement the Bible corpus with contemporary, everyday conversational language, we developed a scraper for `lughayangu.com` (a Kenyan community dictionary).

### Scraping Methodology
1. **Harvester phase**: The sitemap is thin and lacks word URLs. To discover dictionary pages, we queried the global search engine with the top 300 most frequent words from the Bible corpus, harvesting **364 unique Ekegusii word URLs**.
2. **Polite Crawling**: Checks `robots.txt` at runtime, identifies itself with a contact User-Agent, and enforces a rate-limit of 1.5 seconds per request.
3. **Character-trigram Language Identifier**: Learns on `output/bible_en_guz_swh.csv` (achieving 99.5%+ accuracy) to classify blocks as English, Ekegusii, or Kiswahili, filtering out interface text and off-target languages.
4. **Boilerplate Suppression**: Automatically detects and discards blocks appearing across multiple pages (e.g. login buttons, headers, footer notices).
5. **Yield**:
   - `output/lughayangu_sentences.csv`: **316** clean contemporary sentence pairs.
   - `output/lughayangu_lexicon.csv`: **574** vocabulary glosses.
   - `output/lughayangu_unpaired.csv`: **280** unpaired definitions.


