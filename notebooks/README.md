# Fine-tuning notebooks

Run these in order. Each one writes its outputs to `artifacts/` and prints the
name of the next notebook when it finishes.

| # | Notebook | What it does | Runtime |
|---|----------|--------------|---------|
| 01 | `01_eda.ipynb` | Audits every corpus; measures the register gap between scripture and PSAs | ~2 min, CPU |
| 02 | `02_retranslate_swahili.ipynb` | **OPTIONAL — not needed for the current plan.** Regenerates Kiswahili PSA targets with NLLB-1.3B. Only run it if you later add the `eng→swh` direction. | 40–90 min, GPU |
| 03 | `03_build_training_data.ipynb` | Builds the stage-1 / stage-2 / mixed training sets with leakage checks | <1 min |
| 04 | `04_extend_tokenizer.ipynb` | Adds `guz_Latn` to NLLB and initialises it from Kikuyu; measures subword fertility | 2–3 min |
| 05 | `05_finetune.ipynb` | Trains all three models: stage 1, stage 2, and the mixed control | 8–13 h total, GPU |
| 06 | `06_evaluate.ipynb` | Four-way comparison with per-domain breakdown and bootstrap CIs | 45–90 min, GPU |
| 07 | `07_inference_and_export.ipynb` | Translation helper, bulk PSA→Ekegusii draft, post-editing worksheet, export | minutes–2 h |

## The experiment

Three models, identical data and hyperparameters except what is being tested:

| Run | Starts from | Trains on | Purpose |
|---|---|---|---|
| **Stage 1** | tokenizer-extended base | Bible + storybooks + lughayangu (57k) | learn Ekegusii — **the baseline** |
| **Stage 2** | **stage 1** | PSA_KE ×4 + 25% Bible replay (30k) | adapt to PSA register — **the result** |
| **Mixed** | tokenizer-extended base | everything at once (80k) | control: was the ordering worth it? |

Scope is **Ekegusii only**: `eng→guz` and `swh→guz`. Kiswahili appears as a
source language, never a target, so the 50k Kiswahili PSA corpus is unused and
notebook 02 can be skipped.

Notebook 06 computes three numbers that decide what you write up:

- **PSA gain** = stage2 − stage1 on real PSAs → was domain adaptation worth it?
- **Forgetting** = stage1 − stage2 on the Bible → is `REPLAY_FRACTION` high enough?
- **Curriculum benefit** = stage2 − mixed on real PSAs → did the ordering matter?

`nb_common.py` holds paths, the plot palette and small helpers so the notebooks
stay about the modelling.

## Running on the Kinesis GPU node

```bash
git clone https://github.com/SamAbr/public-service-anouncement-MT.git
cd public-service-anouncement-MT
jupyter lab --ip 0.0.0.0 --no-browser
```

Then run **`00_setup.ipynb`** first. It installs dependencies, checks the GPU,
downloads every data file from GitHub and verifies what arrived. Nothing needs
to be uploaded by hand.

Start Jupyter from the repo root so `nb_common.find_project_root()` resolves.

### How data loading works

`nb_common.require_files()` checks for a file locally and, if it is missing,
fetches it from `raw.githubusercontent.com`. So the notebooks run identically on
a laptop with the repo cloned and on a bare GPU node with nothing staged.

The branch matters: `nb_common.GITHUB_BRANCH` is `"main"`, which is this repo's
default branch on GitHub, even though local clones sit on `master`. Push with
`git push origin HEAD:main` (see `push_to_github.sh`) or the raw URLs will 404.

Trained models land in `artifacts/` and are **not** pushed — far too large.
Copy them off the node yourself.

## Hardware notes (A100-SXM4-80GB, 22 vCPU, 118 GB RAM)

- The 600M full fine-tune needs roughly 10 GB for weights, gradients and Adam
  state, so batch size is limited by activations rather than parameters.
- **The node is shared** — five other apps were running when this was written.
  Every GPU notebook prints *free* VRAM at the top; size `BATCH` against that,
  not against the 80 GB headline.
- `bf16` is used rather than `fp16`: native on A100 and avoids fp16 loss-scaling
  instability in seq2seq training.

## Data preparation before notebook 01

```bash
pip install ftfy
python ekegusii/prepare_psa_ke.py
```

This merges `PSA_KE_Final.csv` and `_PSA_EnGuz.csv` into 3,509 training rows and
a 573-row PSA-register test set. Everything downstream depends on it.

## Known limitations

- The PSA Ekegusii references were supplied by the project supervisor; the
  translator and the quality-assurance process are unknown. Report it that way.
- Bible test sets measure whether the model learned Ekegusii, not whether it can
  translate a PSA. Only `psa_ke_heldout` answers the second question.
- 67 test rows and 28 dev rows are dropped automatically because the Bible
  contains 421 duplicate English verses (parallel passages), which would
  otherwise leak the source across the split.
