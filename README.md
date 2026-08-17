# PSA-MT

> **United States International University–Africa** · School of Science and Technology · Natural Language Processing · 2026

Machine translation for Kenyan public service announcements, in two parts.

| Folder | What it is |
|---|---|
| **[`PSA-MT/`](PSA-MT/)** | **The fine-tuning project.** Adding Ekegusii to NLLB-200 by transfer learning and training it on scripture, everyday sentences and Kenyan public service announcements. Models, evaluation, demo and write-ups. |
| **[`corpus_generation/`](corpus_generation/)** | **The corpus pipeline.** Synthesising a large English announcement corpus with a constrained LLM and translating it into Kiswahili, Somali and Dholuo. |

They are separate pipelines with separate dependencies and separate test suites.
Each folder has its own README and its own `requirements.txt`. The only thing
that crosses between them is data: `corpus_generation/data/` holds the
announcements that `PSA-MT/` fine-tunes on.

---

## Start here

**To understand the Ekegusii result**, read [`PSA-MT/README.md`](PSA-MT/README.md).
It covers where all 62,669 training pairs came from, the curriculum experiment,
the metrics, and the limitations.

```bash
cd PSA-MT
pip install -r requirements.txt
```

**To regenerate or extend the announcement corpus**, read
[`corpus_generation/README.md`](corpus_generation/README.md).

```bash
cd corpus_generation
pip install -r requirements.txt
```

---

## Headline result

chrF2++ on real Kenyan public service announcements the model never saw:

| Direction | Stock NLLB-200 | Fine-tuned | Gain |
|---|---:|---:|---:|
| English into Ekegusii | 14.56 | **40.97** | +26.41 |
| Kiswahili into Ekegusii | 14.13 | **39.61** | +25.48 |

Stock NLLB-200 cannot produce Ekegusii at all, so its column is a floor rather
than a baseline. Full context, including why chrF2++ and not BLEU, is in
[`PSA-MT/README.md`](PSA-MT/README.md).

---

## Layout

```
.
├── PSA-MT/                  # fine-tuning: notebooks, training, serving, docs
│   ├── notebooks/  ekegusii/  serve/  docs/  tests/
│   ├── data/                # scripture, everyday sentences, announcements
│   ├── nb_common.py  train_stages.py  MODEL_CARD.md
│   └── requirements.txt
│
├── corpus_generation/       # synthesising and translating the corpus
│   ├── src/  english/  corpus_translation/  tests/
│   ├── data/                # english_psas.csv, psa_parallel_dataset.csv
│   ├── pipeline.ipynb  report.md
│   └── requirements.txt
│
├── push_to_github.sh
└── LICENSE
```

Model weights, checkpoints and figures are written to `artifacts/` inside
whichever project produced them, and are gitignored. 2.4 GB of weights has no
business in a repository with a 100 MB per-file limit.

---

## Team

Weldesenbet Zeray · Samuel Abrha · Hetal Kumbharana · Halima Mohammed ·
Peter Kidiga · Mitchelle Moraa

**Supervisor:** Professor Edward Ombui

---

## License

MIT for the code, see [LICENSE](LICENSE). The corpora are a separate matter:
`PSA-MT/data/bible_en_guz_swh.csv` derives from the Ekegusii Revised Bible,
© Bible Society of Kenya, and the announcement text has not been cleared for
redistribution. **The three fine-tuned model repositories are private and must
stay private** until a supervisor confirms otherwise.
