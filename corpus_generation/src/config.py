import os

# Base directory setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configuration values
DEFAULT_SIZE = 50000
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "psa_parallel_dataset.csv")
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "psa_generation_checkpoint.json")

# Domain definition
DOMAINS = [
    "Education",
    "Agriculture",
    "Security & Safety",
    "Governance",
    "Health"
]

# Random seed for reproducibility
RANDOM_SEED = 42

# Validation boundaries
MIN_WORDS = 25
MAX_WORDS = 60

# NLLB-200 Translation Config
MODEL_NAME = "facebook/nllb-200-distilled-600M"
SRC_LANG = "eng_Latn"
TGT_LANG = "swh_Latn"
BATCH_SIZE = 32
