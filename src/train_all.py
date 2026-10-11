from __future__ import annotations

import argparse
import logging

from src.config import DATASET_NAMES, ProjectConfig
from src.pipeline import run_pipeline
from src.transfer_analysis import run_transfer_analysis

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def train_all_datasets(
    datasets: tuple[str, ...] = DATASET_NAMES,
    *,
    train_lstm: bool = True,
    max_epochs: int = 100,
    include_transfer: bool = False,
) -> dict:
    results = {}
    for dataset in datasets:
        try:
            logger.info("--- Processing %s ---", dataset)
            config = ProjectConfig(dataset=dataset, max_epochs=max_epochs)
            results[dataset] = run_pipeline(dataset, config, train_lstm=train_lstm)
        except Exception:
            logger.exception("Failed to process %s", dataset)

    if include_transfer:
        try:
            config = ProjectConfig(dataset=datasets[0], max_epochs=max_epochs)
            results["transfer"] = run_transfer_analysis(
                datasets,
                config,
                train_lstm=train_lstm,
            )
        except Exception:
            logger.exception("Cross-subset transfer analysis failed")

    logger.info("Requested experiment runs complete")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run C-MAPSS experiments across selected subsets")
    parser.add_argument("--datasets", nargs="+", choices=DATASET_NAMES, default=list(DATASET_NAMES))
    parser.add_argument("--baseline-only", action="store_true", help="Skip TensorFlow and run Ridge only")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--include-transfer", action="store_true", help="Add source-to-target transfer runs")
    args = parser.parse_args()
    train_all_datasets(
        tuple(args.datasets),
        train_lstm=not args.baseline_only,
        max_epochs=args.epochs,
        include_transfer=args.include_transfer,
    )


if __name__ == "__main__":
    main()
