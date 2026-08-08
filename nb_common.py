"""
nb_common.py - shared configuration and helpers for the fine-tuning notebooks.

Imported by every notebook so that paths, plot styling and random seeds are
defined in exactly one place. Keep notebook cells about *what* is being done;
keep the plumbing here.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------

def find_project_root(start: Path | None = None) -> Path:
    """Walk upwards until we find the repo root (the folder containing output/)."""
    here = (start or Path.cwd()).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "output").is_dir() and (candidate / "src").is_dir():
            return candidate
    return here


# ---------------------------------------------------------------------------
# GITHUB
# ---------------------------------------------------------------------------
# The notebooks are designed to run on a bare GPU node. Any data file that is
# missing locally is fetched from the repository over HTTPS, so a node needs
# only this file and the notebook - no manual uploads, no shared filesystem.

GITHUB_USER = "SamAbr"
GITHUB_REPO = "public-service-anouncement-MT"
GITHUB_BRANCH = "main"        # the repo's default branch (local clones use master)
RAW_BASE = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{GITHUB_BRANCH}"

# filename -> path inside the repository
REPO_PATHS = {
    "bible_en_guz_swh.csv": "output/bible_en_guz_swh.csv",
    "psa_ke_train.csv": "output/psa_ke_train.csv",
    "psa_ke_test.csv": "output/psa_ke_test.csv",
    "psa_ke_test_en_guz.csv": "output/psa_ke_test_en_guz.csv",
    "lughayangu_sentences.csv": "output/lughayangu_sentences.csv",
    "english_psas.csv": "output/english_psas.csv",
    "psa_parallel_dataset.csv": "output/psa_parallel_dataset.csv",
    "PSA_KE_Final.csv": "output/PSA_KE_Final.csv",
    "_PSA_EnGuz.csv": "output/_PSA_EnGuz.csv",
}

ROOT = find_project_root()
OUTPUT = ROOT / "output"
ARTIFACTS = ROOT / "artifacts"          # models, tokenizers, checkpoints
DATA = ARTIFACTS / "data"               # training splits
FIGURES = ARTIFACTS / "figures"
for _d in (ARTIFACTS, DATA, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)

# Inputs produced by earlier stages of the project
BIBLE_CSV = OUTPUT / "bible_en_guz_swh.csv"          # en / guz / swh triples
LUGHAYANGU_CSV = OUTPUT / "lughayangu_sentences.csv"  # en / guz pairs
PSA_PARALLEL_CSV = OUTPUT / "psa_parallel_dataset.csv"
ENGLISH_PSA_CSV = OUTPUT / "english_psas.csv"

# Professor-supplied Kenyan PSA corpora, merged by ekegusii/prepare_psa_ke.py
PSA_KE_TRAIN_CSV = OUTPUT / "psa_ke_train.csv"
PSA_KE_TEST_CSV = OUTPUT / "psa_ke_test.csv"

# Produced by the notebooks
PSA_SWH_13B_CSV = OUTPUT / "psa_en_swh_nllb13b.csv"   # notebook 02 (optional)
EXTENDED_MODEL = ARTIFACTS / "nllb600m-guz-init"      # notebook 04

# Notebook 05 trains three models on identical data, for a three-way comparison:
STAGE1_MODEL = ARTIFACTS / "nllb600m-stage1-general"   # Bible + storybooks only
STAGE2_MODEL = ARTIFACTS / "nllb600m-stage2-psa"       # stage 1 -> PSA + replay
MIXED_MODEL = ARTIFACTS / "nllb600m-mixed-control"     # everything at once
FINETUNED_MODEL = STAGE2_MODEL                          # the headline model

# ---------------------------------------------------------------------------
# LANGUAGE CODES
# ---------------------------------------------------------------------------

ENG = "eng_Latn"
SWH = "swh_Latn"
GUZ = "guz_Latn"        # NOT in stock NLLB-200 - added in notebook 04
GUZ_INIT_FROM = "kik_Latn"   # Kikuyu: Kenyan Bantu, nearest available neighbour

BASE_MODEL = "facebook/nllb-200-distilled-600M"
TEACHER_MODEL = "facebook/nllb-200-1.3B"

# Kenyan acronyms that must survive translation intact
ACRONYMS = ["NTSA", "KEPHIS", "HELB", "KUCCPS", "KRA", "CBK", "EACC",
            "SHA", "KEMRI", "KEMSA", "NDMA", "KNEC", "TSC", "ODPC",
            "PPB", "DCI", "NPS", "Huduma", "iTax"]

SEED = 42

# ---------------------------------------------------------------------------
# PLOT STYLE
# ---------------------------------------------------------------------------
# Categorical hues in FIXED order - never cycled, never reordered per chart.
# Assign slot 1 to the first series, slot 2 to the second, and so on, so a
# language keeps its colour across every figure in the series.

PALETTE = ["#2a78d6",  # 1 blue
           "#eb6834",  # 2 orange
           "#1baf7a",  # 3 aqua
           "#eda100",  # 4 yellow
           "#e87ba4",  # 5 magenta
           "#008300",  # 6 green
           "#4a3aa7",  # 7 violet
           "#e34948"]  # 8 red

SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
              "#256abf", "#184f95", "#0d366b"]

# One colour per language, held constant across all notebooks.
LANG_COLOR = {"english": PALETTE[0], "ekegusii": PALETTE[1],
              "swahili": PALETTE[2], "kiswahili": PALETTE[2],
              "somali": PALETTE[3], "luo": PALETTE[4]}

INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e5e4e0"


def use_house_style() -> None:
    """Recessive grid, thin marks, no chartjunk. Call once per notebook."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update({
        "figure.figsize": (9, 4.5),
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "600",
        "axes.titlelocation": "left",
        "axes.titlepad": 12,
        "axes.labelcolor": INK_MUTED,
        "axes.edgecolor": GRID,
        "axes.linewidth": 1.0,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": plt.cycler(color=PALETTE),
        "grid.color": GRID,
        "grid.linewidth": 1.0,
        "text.color": INK,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "lines.linewidth": 2.0,
        "lines.markersize": 8,
    })


def save_fig(fig, name: str) -> Path:
    """Save a figure into artifacts/figures/ and report where it went."""
    path = FIGURES / f"{name}.png"
    fig.savefig(path)
    print(f"saved figure -> {path}")
    return path


# ---------------------------------------------------------------------------
# MISC HELPERS
# ---------------------------------------------------------------------------

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def gpu_report() -> None:
    """Print what hardware we actually have, and how much of it is free."""
    try:
        import torch
    except ImportError:
        print("torch not installed")
        return
    if not torch.cuda.is_available():
        print("NO GPU VISIBLE - training will be unusably slow")
        return
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        total = props.total_memory / 1024 ** 3
        free = torch.cuda.mem_get_info(i)[0] / 1024 ** 3
        print(f"GPU {i}: {props.name}")
        print(f"  total VRAM : {total:6.1f} GB")
        print(f"  free  VRAM : {free:6.1f} GB   <- size batches against THIS,")
        print("                                   the node is shared")
    print(f"\ntorch {torch.__version__}  |  CUDA {torch.version.cuda}")


def download(filename: str, dest: Path | None = None, force: bool = False) -> Path:
    """Fetch one data file from the repository over HTTPS."""
    import urllib.error
    import urllib.request

    if filename not in REPO_PATHS:
        raise KeyError(f"{filename!r} is not in REPO_PATHS - add it there first")
    dest = Path(dest) if dest else ROOT / REPO_PATHS[filename]
    if dest.exists() and not force:
        return dest

    url = f"{RAW_BASE}/{REPO_PATHS[filename]}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {filename} ...", end="", flush=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as fh:
            total = 0
            while chunk := resp.read(1 << 20):
                fh.write(chunk)
                total += len(chunk)
        tmp.replace(dest)
        print(f" {total / 1024 ** 2:.1f} MB")
    except urllib.error.HTTPError as exc:
        tmp.unlink(missing_ok=True)
        raise FileNotFoundError(
            f"\n{url}\nreturned HTTP {exc.code}.\n"
            f"Either the file has not been pushed yet, or GITHUB_BRANCH "
            f"({GITHUB_BRANCH!r}) is wrong for this repository."
        ) from None
    return dest


def require_files(*paths, allow_download: bool = True) -> None:
    """
    Verify inputs exist, fetching them from GitHub when they do not.

    This is what lets a fresh GPU node run the notebooks with nothing staged on
    disk: anything listed in REPO_PATHS is pulled on demand.
    """
    for p in paths:
        path = Path(p)
        if not path.exists() and allow_download and path.name in REPO_PATHS:
            download(path.name, path)
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required input: {path}\n"
                + ("It is not in REPO_PATHS, so it cannot be fetched - "
                   "run the notebook that produces it."
                   if path.name not in REPO_PATHS else
                   "Download failed; check network access from this node.")
            )
        size = path.stat().st_size / 1024 ** 2
        print(f"  ok  {path.name:<34} {size:8.1f} MB")


def save_json(obj, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)
    print(f"wrote {path}")


def load_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def banner(title: str) -> None:
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)
