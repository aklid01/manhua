"""Manhua translation pipeline package."""

import os
import warnings

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TQDM_DISABLE"] = "1"
os.environ["PYTHONWARNINGS"] = "ignore"

warnings.filterwarnings("ignore")

import logging

for _log_name in [
    "transformers",
    "huggingface_hub",
    "huggingface_hub.file_download",
    "filelock",
    "urllib3",
    "paddle",
    "ppocr",
]:
    logging.getLogger(_log_name).setLevel(logging.ERROR)

try:
    from huggingface_hub.utils import disable_progress_bars

    disable_progress_bars()
except Exception:
    pass

__version__ = "0.1.0"
