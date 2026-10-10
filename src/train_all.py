import logging
from src.pipeline import run_pipeline

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def train_all_datasets():
    datasets = ["FD001", "FD002", "FD003", "FD004"]
    all_results = {}

    for ds in datasets:
        try:
            logger.info(f"--- Processing Dataset {ds} ---")
            results = run_pipeline(ds)
            all_results[ds] = results
        except Exception as e:
            logger.error(f"Failed to process {ds}: {e}")

    logger.info("Training complete for all datasets.")
    return all_results

if __name__ == "__main__":
    train_all_datasets()
