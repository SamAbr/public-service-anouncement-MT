# `serve/` — the four-system comparison UI

A small FastAPI service and a single-page front end that runs the same four
systems `notebooks/06_evaluate.ipynb` scores, side by side, on whatever sentence
a visitor types.

The point is not "here is a translator." The point is that the paper's central
claim — that a second, PSA-specific stage buys **+10.97 chrF2++** over stage 1 —
is a number on a table until someone types *"HELB loan applications close on 30
September"* and reads four outputs next to each other. Then it is visible.

| System | What it is |
|---|---|
| `stock` | `facebook/nllb-200-distilled-600M`, asked for `kik_Latn` |
| `stage1` | fine-tuned on the Bible and storybook corpus |
| `stage2` | stage 1, then adapted on Kenyan PSAs with Bible replay — **the headline model** |
| `mixed` | the control: all the same data in one pass, no curriculum |

**On `stock`.** Ekegusii has no token in stock NLLB-200, so it *cannot* be asked
for Ekegusii. Following notebook 06, it is asked for Kikuyu (`kik_Latn`), the
nearest Kenyan Bantu language it does support. That establishes what "no
Ekegusii support" looks like numerically. It is a floor, not a fair comparison,
and the UI says so on the card. Do not let a slide turn it into one.

---

## Run it

### 1. A read token

The three fine-tuned repositories are private. Create a **read**-scoped token at
<https://huggingface.co/settings/tokens>. Not a write token — this service never
uploads anything, and a leaked write token can delete your models.

```bash
cp .env.example .env
# put the token in .env; .env is gitignored, keep it that way
```

### 2. Docker

```bash
docker compose --profile gpu up --build     # a box with an NVIDIA card
docker compose --profile cpu up --build     # anything else
```

Then open <http://localhost:8000>.

The first start downloads ~7 GB of weights. They land in the `hf-cache` volume,
so every later start is fast — don't delete that volume casually.

### 3. Without Docker

```bash
pip install -r requirements.txt
export HF_TOKEN=hf_...
uvicorn app:app --host 0.0.0.0 --port 8000
```

### 4. Front-end work with no weights at all

```bash
MOCK_MODE=1 uvicorn app:app --port 8000
```

Every system returns a tagged placeholder. Nothing is downloaded, nothing needs
a GPU, and the whole UI — cards, chart, table view, both themes — is exercised.

---

## What it needs from the machine

| | GPU | CPU |
|---|---|---|
| Memory for four models | ~4.8 GB VRAM (fp16) | ~9.6 GB RAM (fp32) |
| Per sentence, per system, beam 4 | ~0.3 s | ~3–8 s |

If the host has less memory than that, set `MAX_RESIDENT=2`. Models are then
evicted least-recently-used and reloaded on demand: identical output, a pause
when a visitor switches systems. `BEAMS=1` roughly halves CPU latency at a real
cost in quality — only worth it on a CPU-only host.

`BEAMS=4` is the default because it is what notebook 06 evaluated with, so a
number quoted in the paper and a string shown on screen came out of the same
settings. Change it and they no longer correspond.

---

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | the UI |
| `GET` | `/api/health` | liveness, device, which models are resident |
| `GET` | `/api/systems` | system metadata + chrF scores + runtime info |
| `GET` | `/api/metrics` | the parsed `evaluation_results.csv` |
| `POST` | `/api/translate` | `{text, src_lang, systems[], beams, max_new_tokens}` |

The front end sends **one request per system, in parallel**, so each card
resolves the moment its own model finishes instead of every card waiting for the
slowest.

```bash
curl -s localhost:8000/api/translate \
  -H 'Content-Type: application/json' \
  -d '{"text":"Report suspected cholera cases to the nearest health facility.",
       "src_lang":"eng_Latn","systems":["stage1","stage2"]}' | python -m json.tool
```

### Paragraphs

NLLB is a sentence-level model; handing it a whole paragraph makes it drop
clauses. `segment()` splits on line breaks and sentence boundaries, translates
each piece, and rejoins — preserving blank lines, because a PSA's shape carries
meaning. `MAX_SEGMENTS` (default 40) caps how much one request may ask for.

---

## Keeping the chart honest

The chart reads `metrics/evaluation_results.csv`, falling back to
`../artifacts/data/evaluation_results.csv`. Whichever columns exist are the
columns it draws — so until `06_evaluate.ipynb` has been re-run with all three
checkpoints present, the mixed control is absent from the chart and the UI says
so in a notice rather than quietly showing a three-bar chart as if it were
complete.

**After re-running notebook 06, copy the new CSV over `metrics/` and rebuild.**

---

## Before this is public

The weights derive from the Ekegusii Revised Bible (© Bible Society of Kenya)
and PSA data of unconfirmed provenance. The repositories are private for that
reason. Clear publication with Prof. Ombui before putting this on a public
hostname, and keep the footer's post-editing warning: an untouched machine
translation of a health or safety notice is not something to publish.
