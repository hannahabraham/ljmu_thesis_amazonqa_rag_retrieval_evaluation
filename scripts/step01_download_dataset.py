"""Download AmazonQA train, validation, and test JSONL files.

The script downloads files into ``datasets/raw`` and is idempotent: existing
files are not downloaded again.
"""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

from config.settings import DATASET_FILES, DATASET_URLS
from src.utils.logging_config import configure_logging

configure_logging()
LOGGER = logging.getLogger(__name__)


def _download_http(url: str, destination: Path) -> None:
    """Download a file from an HTTP URL to the destination path."""
    LOGGER.info("Downloading %s -> %s", url, destination)

    with urllib.request.urlopen(url) as response:
        destination.write_bytes(response.read())


def _download_drive(file_id: str, destination: Path) -> None:
    """Download a Google Drive file using gdown."""
    import gdown  # pylint: disable=import-outside-toplevel

    LOGGER.info("Downloading Google Drive file %s -> %s", file_id, destination)

    gdown.download(
        id=file_id,
        output=str(destination),
        quiet=False,
    )


def main() -> None:
    """Download missing AmazonQA dataset files."""
    if not DATASET_FILES["train"].exists():
        _download_http(DATASET_URLS["train"], DATASET_FILES["train"])
    else:
        LOGGER.info("train exists, skipping")

    if not DATASET_FILES["val"].exists():
        _download_http(DATASET_URLS["val"], DATASET_FILES["val"])
    else:
        LOGGER.info("val exists, skipping")

    if not DATASET_FILES["test"].exists():
        _download_drive(
            DATASET_URLS["test_drive_id"],
            DATASET_FILES["test"],
        )
    else:
        LOGGER.info("test exists, skipping")


if __name__ == "__main__":
    main()
