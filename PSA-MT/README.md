# Fine-Tuning Neural Machine Translation Models for Kenyan Public Service Announcements Using a Synthetic Parallel Corpus

> **United States International University–Africa** · School of Science and Technology · Natural Language Processing · 2026

Adding **Ekegusii** to NLLB-200, a language the model was never trained on, so
that Kenyan public service announcements can reach 2.7 million more speakers.

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

## 📌 What this is

NLLB-200 supports 200 languages. Ekegusii, a Bantu language of about 2.7 million
speakers in Kisii and Nyamira counties, is not one of them, and no amount of
prompting will make a model produce a language it has no token for.

So we added one. Using **transfer learning** we registered `guz_Latn` in
NLLB-200, seeded its embedding from a related language the model already knows,
and fine-tuned on **62,669 parallel sentence pairs** collected and built for this
project. The result translates English and Kiswahili into Ekegusii.

Two things are documented here, because they are the two things the project
actually decided:

1. **[Where the training data came from](#-data-collection)** and what had to be
   thrown away.
2. **[The curriculum experiment](#-the-curriculum-experiment)**: whether teaching
   the model general Ekegusii first and public service register second beats
   teaching it both at once.

### Headline result

chrF2++ on **real** Kenyan public service announcements the model never saw:

| Direction | Stock NLLB-200 | **Our model** | Gain |
|---|---:|---:|---:|
| English into Ekegusii | 14.56 | **40.97** | +26.41, a 181% relative gain |
| Kiswahili into Ekegusii | 14.13 | **39.61** | +25.48, a 180% relative gain |

Stock NLLB-200 cannot produce Ekegusii at all. It was asked for the nearest
language it supports, so its column is a **floor**, not a baseline. Nobody
should read it as a fair competitor.

---

## 📥 Data collection

Nothing off the shelf exists for English into Ekegusii, so the corpus was built
from three sources, each collected a different way.

### 1. Ekegusii Bible, 56,866 pairs

The Ekegusii Revised Bible aligned verse by verse against English and Kiswahili
translations of the same verses, from eBible USFM sources.

`ekegusii/build_trilingual_corpus.py` parses the USFM markup, keys every verse by
its canonical book, chapter and verse reference, and emits a row only where all
three languages have text for that exact reference. Verses missing in any one
language are dropped rather than approximately matched, because a misaligned
pair teaches the model a wrong mapping and there is no way to detect that later.

This is the only large, genuinely parallel, human-translated Ekegusii text that
was available to us. It is also the source of the project's biggest limitation:
see [Limitations](#-limitations).

### 2. Kenyan public service announcements, 5,692 pairs

The domain the project is actually about, and the part that had to be
manufactured.

```
Scenario knowledge base  (Health, Education, Agriculture, Governance, Safety)
         │
         ▼
   Metadata planner          domain-balanced permutation of scenario × entity × action
         │
         ▼
 GPT-5 mini (Azure OpenAI)   constrained few-shot synthesis, 10 parallel workers
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
   Validated English public service announcements
         │
         ├──► Kiswahili   via NLLB-200
         └──► Ekegusii    via GPT-5 mini, few-shot prompted with hand-curated
                          Ekegusii seed pairs (../corpus_generation/src/knowledge/ekegusii_fewshot_corpus.py)
```

Announcements that failed any validation rule were rewritten and re-checked
rather than discarded, which is why the generator has a rewrite loop rather than
a simple filter.

**These Ekegusii references are machine generated.** They are the weakest link in
the corpus and are flagged as such throughout. Because the PSA portion is only
9.1% of the data yet is the entire target domain, those 5,692 pairs are
**upsampled ×4** during training so scripture does not swamp them.

### 3. Lughayangu everyday sentences, 111 pairs

`ekegusii/scrape_lughayangu.py` collects contemporary English and Ekegusii
sentence pairs from lughayangu.com. Small, but the only modern, human-written,
non-scriptural Ekegusii in the corpus, and the only thing standing between the
model and a purely biblical register.

### What went in

Each aligned record yields up to two directional pairs: English→Ekegusii, and
Kiswahili→Ekegusii wherever a Kiswahili column exists.

| Source | Direction | Pairs |
|--------|-----------|------:|
| Ekegusii Bible | English to Ekegusii | 28,439 |
| Ekegusii Bible | Kiswahili to Ekegusii | 28,482 |
| Lughayangu everyday sentences | English to Ekegusii | 111 |
| Duplicates removed | | -55 |
| Kenyan public service announcements | English to Ekegusii | 3,509 |
| Kenyan public service announcements | Kiswahili to Ekegusii | 2,183 |
| **Total unique training pairs** | | **62,669** |

By source, ignoring direction:

| Source | Pairs | Share |
|--------|------:|------:|
| Ekegusii Bible | 56,866 | 90.7% |
| Kenyan public service announcements | 5,692 | 9.1% |
| Everyday sentences | 111 | 0.2% |

**Held out and never trained on:** 2,993 scripture pairs, 200 everyday
sentences, 944 public service announcements. Splits are by source, so the test
sets measure three different things rather than averaging them into one number
that describes nothing.

The model sees **79,745 examples per epoch** rather than 62,669, because of the
×4 upsampling described above. That is a training weight, not extra data, and it
is why an earlier draft of the poster quoting "80,000 records" was wrong.

`docs/count_records.py` recomputes every figure in this section from the CSVs
in `data/`, so none of them are copied by hand.

### What was collected and then thrown away

Two sources contributed **nothing** to the released model, which is worth stating
plainly because both appear in the repository and would otherwise be assumed to
be in the training set:

- **4laws**, 27,575 pairs. Every one a duplicate of text already present in the
  eBible corpus. Deduplication removed all of them.
- **African Storybook**, 110 rows. Merged into the Bible CSV *after* training
  finished. The released weights never saw them. This is provable rather than
  asserted: the trained model's corpus had 30,753 Bible rows expanding to 56,977
  pairs, and the CSV in the repository today has 30,863 rows expanding to 57,160.

---

## 🔧 Method

`guz_Latn` was added to the tokenizer and its embedding **seeded from `kik_Latn`
(Kikuyu)** plus 1% noise rather than initialised randomly, so the new language
starts from what NLLB already knows about a related Bantu language instead of
from nothing. The noise stops the two tokens being exact twins.

NLLB expects the source sequence as `[src_lang] … [eos]` and is steered to the
target by `forced_bos_token_id`. The training script and `serve/app.py` both
build that framing by hand rather than trusting tokenizer defaults, because a
tokenizer that does not know `guz_Latn` silently maps it to `<unk>` and the model
then decodes into whatever language it likes. `tests/verify_hf_uploads.py`
asserts `guz_Latn != unk` on every published repository for exactly that reason.

Base model: `facebook/nllb-200-distilled-600M`.

---

## 🧪 The curriculum experiment

The question: for a language the model has never seen, is it better to teach
general Ekegusii first and the public service register second, or to teach both
at once?

| System | What it is |
|---|---|
| **Stage 1** | General Ekegusii only. Scripture and everyday sentences, no PSAs. |
| **Two-stage** | Stage 1, then a second pass on PSAs with replay of earlier data. |
| **Single pass** | One pass over everything mixed together, PSAs upsampled ×4. |

Both curricula see the same unique examples. The two-stage run took about **9%
more gradient updates** than the single pass, so it was not undertrained.

### chrF2++

| Direction | Test set | n | Stock | Stage 1 | Two-stage | **Single pass** |
|---|---|---:|---:|---:|---:|---:|
| eng→guz | Real PSAs | 570 | 14.56 | 23.96 | 34.93 | **40.97** |
| eng→guz | Scripture | 1,459 | 14.88 | 49.66 | 48.95 | **49.81** |
| eng→guz | Everyday prose | 200 | 11.41 | 28.97 | 29.37 | **32.98** |
| swh→guz | Real PSAs | 371 | 14.13 | 24.49 | 34.50 | **39.61** |
| swh→guz | Scripture | 1,463 | 14.65 | 49.11 | 48.80 | **49.30** |

### BLEU, same runs

| Direction | Test set | n | Stock | Stage 1 | Two-stage | **Single pass** |
|---|---|---:|---:|---:|---:|---:|
| eng→guz | Real PSAs | 570 | 1.63 | 2.54 | 7.21 | **12.33** |
| eng→guz | Scripture | 1,459 | 0.83 | 19.75 | 19.33 | **19.91** |
| eng→guz | Everyday prose | 200 | 0.07 | 1.27 | 1.21 | **1.59** |
| swh→guz | Real PSAs | 371 | 1.01 | 1.77 | 6.87 | **10.66** |
| swh→guz | Scripture | 1,463 | 1.20 | 19.79 | 19.79 | **19.97** |

### Finding: the curriculum lost

**The single pass beat the two-stage curriculum on every test set**, by 6.04
chrF2++ on English PSAs and 5.11 on Kiswahili PSAs, so the single-pass model is
the one we release. The curriculum checkpoint is published as the ablation.

Two obvious objections, both ruled out. It did not see less data: the unique
examples are identical. It was not trained less: it took more gradient updates.

The mechanism is visible in the loss curve. The two-stage run's dev loss bottoms
out at **1.357** and then drifts back up to **1.417** during the second stage.
Narrowing to PSAs at the end, even with replay, overfits a 5,692-pair domain
faster than it specialises to it. Mixing keeps the general signal present in
every batch instead of withdrawing it and hoping replay is enough.

Notably the two-stage run is *not* worse at scripture either, so this is not
classic catastrophic forgetting. It is plain overfitting on the small final
domain.

### Why chrF2++ is the headline metric and BLEU the footnote

Ekegusii is agglutinative: one word carries what English spreads across four or
five. BLEU counts whole-word n-gram matches, so a translation that gets the stem
and every prefix right but misses one final suffix scores exactly the same as a
completely wrong word, namely zero.

chrF2++ (Popović) compares character n-grams, so a nearly correct word form earns
partial credit, and it correlates far better with human judgement on Bantu
languages. The everyday-prose row makes the case on its own: **1.59 BLEU** reads
as total failure while **32.98 chrF2++** reflects output a speaker can actually
read. Both come from the same sentences.

COMET, the usual third option, has no Ekegusii support and could not be used at
all. Both metrics are computed with sacreBLEU so the numbers are comparable to
published work.

Raw figures: `docs/metrics_full.json`, `docs/bleu_three_systems.json`.

---

## 🤖 Models

| Repository | What it is |
|---|---|
| `samuelabrha/nllb-200-600M-ekegusii-mixed` | **Released model**, single pass |
| `samuelabrha/nllb-200-600M-ekegusii-stage1` | Baseline, general Ekegusii only |
| `samuelabrha/nllb-200-600M-ekegusii-psa` | Two-stage curriculum, the ablation |

All three are **private** pending licence clearance. See [License](#-license).

Three checkpoints fine-tuned from one base look alike, so which weights landed in
which repository was verified rather than assumed:

```bash
python tests/verify_hf_uploads.py     # every repo loads and guz_Latn is a real token
python tests/identify_hf_models.py    # weight fingerprint: which repo holds what
```

`identify_hf_models.py` takes float64 cosine similarity over fixed tensor slices
plus the added embedding row. Every repository matched its own local directory at
**1.000000** and every cross pair fell to **0.99985** or below, so nothing is
mislabelled. The margin is real but tighter than designed, which is expected for
siblings fine-tuned from a shared base; behavioural rescoring is the tiebreaker
and agrees.

---

## 🚀 Running it

Everything below is run from inside the `PSA-MT/` folder.

```bash
cd PSA-MT
pip install -r requirements.txt
```

### Pipeline

```
00_setup → 01_eda → 02_build_training_data → 03_extend_tokenizer
        → train_stages.py → 04_evaluate → 05_inference_and_export
```

Training is a script rather than a notebook because it has to survive a shared
GPU: it waits for free VRAM before starting, recovers from out-of-memory by
halving the batch size and retrying, and can resume a single stage without
repeating the others.

### Demo

```bash
bash serve/run_public_demo.sh     # GPU node, public URL via Cloudflare tunnel
serve/colab_demo.ipynb            # free Colab T4, four cells, no local GPU needed
```

The public demo serves the released model only, with a dropdown for
English→Ekegusii and Kiswahili→Ekegusii. Set `ENABLED_SYSTEMS` to expose the
comparison view with all four systems side by side:

```bash
ENABLED_SYSTEMS=stock,stage1,stage2,mixed bash serve/run_public_demo.sh
```

The repositories are private, so the service needs `HF_TOKEN` set to a
**read**-scoped token. Never give it a write token. The script refuses to start
if something is already listening on its port: an earlier version happily
health-checked a stale process and silently ignored the new configuration, which
cost an afternoon.

### Tests

```bash
python -m unittest discover tests
python tests/verify_hf_uploads.py     # published models load and tokenise guz_Latn
python tests/identify_hf_models.py    # weight fingerprinting
```

---

## 🗂️ Project layout

This project sits in the `PSA-MT/` folder of the repository. The synthetic
corpus generator that produced the announcement data lives beside it in
`corpus_generation/`, and is a separate pipeline with its own README.

```
PSA-MT/
│
├── notebooks/                      # 00_setup .. 05_inference_and_export
├── nb_common.py                    # paths, seeds, plot style, data downloader
├── train_stages.py                 # Training driver: waits for VRAM, OOM-safe, resumable
│
├── ekegusii/
│   ├── build_trilingual_corpus.py  # English·Ekegusii·Kiswahili verse aligner (eBible USFM)
│   ├── scrape_lughayangu.py        # Contemporary English-Ekegusii pairs (lughayangu.com)
│   ├── scrape_ekegusii_corpus.py   # Wider web collection pass
│   └── prepare_psa_ke.py           # Merges the Kenyan announcement corpora, makes splits
│
├── data/                           # everything this project trains and tests on
│   ├── bible_en_guz_swh.csv        # Aligned English·Ekegusii·Kiswahili triples
│   ├── lughayangu_sentences.csv    # Contemporary sentence pairs
│   ├── PSA_KE_Final.csv            # Kenyan announcements, five languages
│   ├── psa_ke_train.csv            # Splits used by the notebooks
│   ├── psa_ke_test.csv
│   ├── psa_ke_test_en_guz.csv
│   └── psa_ke_manifest.json
│
├── serve/
│   ├── app.py                      # FastAPI translation service
│   ├── static/index.html           # Single page UI, public and comparison modes
│   ├── run_public_demo.sh          # Launch plus Cloudflare tunnel
│   └── colab_demo.ipynb            # Free T4 fallback
│
├── tests/
│   ├── verify_hf_uploads.py        # Published models load, guz_Latn is real
│   ├── identify_hf_models.py       # Weight fingerprinting
│   ├── test_aligner.py             # Offline tests for the verse aligner
│   └── test_lughayangu.py          # Offline tests for the scraper
│
├── docs/
│   ├── IMPLEMENTATION_REPORT.md/.pdf
│   ├── PRESENTATION_BRIEF.md/.pdf
│   ├── Ekegusii_NMT_presentation.pptx
│   ├── banner.pdf/.png/.html       # A1 landscape print poster
│   ├── count_records.py            # Recomputes the corpus tables from data/
│   ├── gen_deck.js, gen_banner.py, gen_brief_pdf.py
│   └── metrics_full.json, bleu_three_systems.json
│
├── MODEL_CARD.md
└── requirements.txt
```

Model weights, tokenizers and figures are written to `artifacts/`, which is
gitignored: 2.4 GB of weights has no business in a repository with a 100 MB
per-file limit.

---

## 📄 Documentation

| Document | For |
|---|---|
| `docs/IMPLEMENTATION_REPORT.pdf` | The team: what was built and why |
| `docs/PRESENTATION_BRIEF.pdf` | Presenting it, including likely questions |
| `docs/Ekegusii_NMT_presentation.pptx` | The slide deck |
| `docs/banner.pdf` | Print poster, A1 landscape, 841 × 594 mm |
| `docs/metrics_full.json` | Every chrF2++ and BLEU figure quoted above |

Each document is generated by a script beside it, so a corrected number can be
pushed through every deliverable instead of edited by hand in four places:

```bash
python docs/count_records.py     # recompute the corpus tables
node   docs/gen_deck.js          # rebuild the slide deck
python docs/gen_banner.py        # rebuild the A1 poster
python docs/gen_brief_pdf.py IMPLEMENTATION_REPORT.md "Implementation report" "subtitle"
```

---

## ⚠️ Limitations

State these before anyone else does.

- **The corpus is 90.7% scripture.** Nine tenths of what the model knows about
  Ekegusii comes from the Bible, which is archaic in register and narrow in
  subject matter. This, not model size, is the binding constraint on quality.
- **The Ekegusii public service announcements are machine generated.** They were
  produced by a prompted LLM, not written or checked by Ekegusii speakers. The
  model is therefore partly learning another model's Ekegusii.
- **No human evaluation.** chrF2++ is a proxy. No speaker has yet rated a single
  output, so quality claims are automatic-metric claims only.
- **Everyday prose is the weakest direction**, at 32.98 chrF2++, which is exactly
  what a 111-pair contemporary sample predicts.
- **Stock NLLB-200 is a floor, not a baseline.** It cannot produce Ekegusii, so
  the +181% figure measures the distance from nothing, not from a real competitor.
- **Intended use is drafting, not publishing.** A county officer writes the
  notice in English, the model drafts the Ekegusii, and a speaker corrects it
  before anything goes out.

---

## 🔭 Future work

- **Rebalance away from scripture.** A few thousand contemporary Ekegusii pairs
  would likely buy more than any architectural change.
- **Replace the synthetic Ekegusii references** with speaker-written or
  speaker-corrected announcements, and measure how much the synthetic ones cost.
- **Human evaluation** with Ekegusii speakers on the held-out announcements.
- Extend the same transfer-learning recipe to other unsupported Kenyan languages.
- Durable hosting for the demo. It currently runs on a shared GPU node that is
  reclaimed daily, with the Colab notebook as the fallback.

---

## 📄 License

This project is for academic research at **USIU–Africa**. Code and the
collection scripts are published to support future research in machine
translation for under-resourced African languages.

**The three model repositories are private and must stay private until a
supervisor confirms otherwise.** The weights derive from the Ekegusii Revised
Bible, © Bible Society of Kenya, and from announcement text whose provenance has
not been cleared for redistribution. Publishing the weights republishes the
training data in compressed form, which is a licensing question rather than a
technical one.
