---
license: cc-by-nc-4.0
language:
  - guz
  - en
  - sw
base_model: facebook/nllb-200-distilled-600M
pipeline_tag: translation
tags:
  - translation
  - ekegusii
  - kisii
  - low-resource
  - kenya
  - nllb
---

# NLLB-200 600M fine-tuned for Ekegusii (guz_Latn)

Adds **Ekegusii** — a Kenyan Bantu language spoken by roughly 2.7 million people —
to NLLB-200, which does not support it, and adapts the model to the register of
Kenyan **public service announcements**.

Supported directions: `eng_Latn → guz_Latn`, `swh_Latn → guz_Latn`.

## Results (chrF2++)

| Direction | Test set | n | Stock NLLB* | Stage 1 | **Stage 2** |
|---|---|---|---|---|---|
| eng→guz | Bible (held out) | 1,459 | 14.88 | 49.66 | 48.95 |
| eng→guz | Contemporary prose | 200 | 11.41 | 28.97 | 29.37 |
| eng→guz | **Real PSAs** | 570 | 14.56 | 23.96 | **34.93** |
| swh→guz | Bible (held out) | 1,463 | 14.65 | 49.11 | 48.80 |
| swh→guz | **Real PSAs** | 371 | 14.13 | 24.49 | **34.50** |

\* Stock NLLB-200 cannot produce Ekegusii; the baseline asks it for `kik_Latn`
(Kikuyu), the nearest supported language. It is a floor, not a fair comparison.

Domain adaptation gained **+10.97 chrF2++ (+45.8%)** on real PSAs while losing
only **0.71** on scripture — the replay mixture prevented catastrophic
forgetting.

## How it was built

`guz_Latn` was added to the tokenizer and its embedding initialised from
`kik_Latn` (Kikuyu, the closest Kenyan Bantu language NLLB supports) rather than
randomly, then trained in two stages:

1. **General Ekegusii** — 56,977 examples: Ekegusii Bible verses aligned to the
   Berean Standard Bible and Neno Kiswahili, plus African Storybook pages and
   contemporary sentence pairs.
2. **PSA adaptation** — 30,357 examples: real Kenyan public service
   announcements (×4) with **25% stage-1 replay** to anchor general competence.

Full fine-tune, 8-bit AdamW, effective batch 48, lr 5e-5 then 1.5e-5.

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tok = AutoTokenizer.from_pretrained("SamuelAbrhaCDLI2025/nllb-200-600M-ekegusii-psa")
model = AutoModelForSeq2SeqLM.from_pretrained("SamuelAbrhaCDLI2025/nllb-200-600M-ekegusii-psa")

text = "Report suspected cholera cases to the nearest health facility."
ids = ([tok.convert_tokens_to_ids("eng_Latn")]
       + tok(text, add_special_tokens=False)["input_ids"]
       + [tok.eos_token_id])

import torch
out = model.generate(input_ids=torch.tensor([ids]),
                     forced_bos_token_id=tok.convert_tokens_to_ids("guz_Latn"),
                     max_new_tokens=128, num_beams=4)
print(tok.batch_decode(out, skip_special_tokens=True)[0])
```

Special tokens are built explicitly (`[lang_code] … [eos]`) rather than via
`tokenizer.src_lang`, because the tokenizer's language machinery does not know
about a language added after pretraining.

## Limitations

- **The Ekegusii references were not verified by the authors.** The PSA corpus
  was supplied by the project supervisor; the translator and the
  quality-assurance process are unknown. No inter-annotator agreement exists.
- **Most training data is scripture.** ~57k of ~87k examples are Bible verses,
  so the model may still lean formal on unfamiliar input.
- **Institutional vocabulary is thin.** 53.6% of English content-word types in
  the PSA corpus never appear in the aligned training data — portals, bursaries,
  agency names. Expect weak handling of `HELB`, `KUCCPS`, `iTax` and similar.
- **No human evaluation.** All numbers are automatic metrics against a single
  reference.
- Not suitable for medical, legal or emergency communication without review by a
  fluent Ekegusii speaker.

## Data licensing

Derived from the Ekegusii Revised Bible (© Bible Society of Kenya, eBible.org),
the Berean Standard Bible (public domain), Neno: Bibilia Takatifu (Biblica open
licence), African Storybook (CC-BY), and PSA corpora supplied by USIU-Africa.
Verify redistribution terms before commercial use.

## Citation

Fine-Tuning Neural Machine Translation Models for Kenyan Public Service
Announcements. United States International University–Africa, Department of
Computing, 2026.
