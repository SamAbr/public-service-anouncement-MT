# Corpus generation

Synthesising a large English corpus of Kenyan public service announcements with
a constrained LLM, then translating it into Kiswahili, Somali and Dholuo.

This is one half of [PSA-MT](../README.md). The other half, `../PSA-MT/`, fine-tunes
NLLB-200 on Ekegusii and consumes the announcements produced here.

```bash
cd corpus_generation
pip install -r requirements.txt
```

---

## Why synthesise

Kenyan public service announcements exist, but not in the volume or the language
coverage a translation model needs, and not with permission to redistribute.
Generating them makes the register controllable: every sentence can be forced to
be a single clear public instruction rather than a press release or a Gazette
notice, which is what the downstream model has to learn.

The cost is honesty about provenance. This corpus is machine written, and the
Ekegusii side of it is machine translated. Both facts are stated in the
fine-tuning project's limitations, and they are the first thing to fix with
speaker-written text.

---

## Pipeline

```
Scenario knowledge base  (Health, Education, Agriculture, Governance, Safety)
         │
         ▼
   Metadata planner        domain-balanced permutation of scenario × entity × action
         │
         ▼
 GPT-5 mini (Azure OpenAI) constrained few-shot synthesis, 10 parallel workers
         │
         ▼
   Validation pipeline
   ├── single sentence, 10 to 25 words
   ├── exactly one clear public action
   ├── public advisory register
   ├── no Gazette or press-release language
   └── Jaccard near-duplicate filtering
         │
         ▼
   data/english_psas.csv
         │
         ├──► Kiswahili   nllb-200-distilled-600M
         ├──► Somali      nllb-200-1.3B
         └──► Dholuo      nllb-200-1.3B
                  │
                  ▼
         data/psa_parallel_dataset.csv
```

A sentence that fails validation is **rewritten and re-checked**, not discarded,
which is why `src/llm_generator.py` carries a rewrite loop rather than a plain
filter. Acronyms that must survive translation intact (`SHA`, `HELB`, `NTSA`,
`KRA`, `Huduma`) are registered as custom tokenizer tokens before translation.

Translation runs on a Colab GPU and checkpoints to Google Drive, so a runtime
disconnect costs minutes rather than the whole run.

---

## Running it

You need an **Azure AI Foundry key** with a GPT-5 mini deployment, and a Google
account for the GPU.

Locally, English generation only:

```bash
python english/generate_english_only.py --size 50000 \
       --output data/english_psas.csv --engine azure_llm --azure-key YOUR_KEY
```

End to end, including the GPU translation step: open `pipeline.ipynb` in Google
Colab, set the runtime to a T4 GPU, and run the cells in order. It clones this
repository, installs dependencies, generates the English corpus, mounts Drive
for checkpointing, and translates.

Never paste the Azure key into a file you commit. Pass it as a notebook
parameter or an environment variable.

---

## Layout

```
corpus_generation/
│
├── english/
│   └── generate_english_only.py    # English synthesis entry point
│
├── corpus_translation/
│   └── translate_colab.py          # GPU translation into Kiswahili, Somali, Dholuo
│
├── src/
│   ├── generator.py                # Concurrent generation orchestrator
│   ├── llm_generator.py            # Azure OpenAI client and rewrite loop
│   ├── validator.py                # Length and register constraints
│   ├── deduplicator.py             # Jaccard near-duplicate filtering
│   ├── exporter.py                 # Domain-interleaved row export
│   ├── retriever.py                # Few-shot example selection
│   ├── translator.py               # NLLB wrapper
│   ├── templates/                  # Per-domain sentence templates
│   ├── knowledge/                  # Scenarios, entities, institutions, hazards,
│   │                               #   locations, audiences, terminology, and the
│   │                               #   hand-curated Ekegusii few-shot seed pairs
│   ├── analysis/eda_analysis.py    # Corpus statistics and plots
│   └── model/train_mt.py           # Early fine-tuning experiment, superseded by ../PSA-MT
│
├── tests/
│   ├── test_generator.py           # Unit tests for generation and validation
│   └── verify_dataset.py           # Columns, formats and nulls in the CSVs
│
├── data/
│   ├── english_psas.csv            # Validated English announcements
│   ├── psa_parallel_dataset.csv    # Four-way parallel corpus
│   └── corpus_manifest.json        # Source IDs, counts, filter statistics
│
├── pipeline.ipynb                  # Colab: generate then translate
├── report.md                       # Full methodology report
└── requirements.txt
```

Plots and any model output land in `artifacts/`, which is gitignored.

---

## Tests

```bash
python -m unittest discover tests
python tests/verify_dataset.py
```

---

## A note on `src/model/train_mt.py`

This was the project's first fine-tuning attempt, training NLLB on the parallel
corpus directly. It is kept because it is part of the record, but it is **not**
what produced the released models. That work moved to `../PSA-MT/`, where adding
Ekegusii to the tokenizer made a different training driver necessary. Read
`../PSA-MT/README.md` for the results, and do not quote numbers from here.
