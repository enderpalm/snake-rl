import csv
from pathlib import Path

METRIC_PATH = "../artifacts/metrics/"

def save_metrics(logs: list[dict], filepath: str) -> None:
    """Save a list of log dictionaries to a CSV file.

    Args:
        logs: List of dictionaries containing metric data
        filepath: Path where the CSV file will be saved
    """
    if not logs:
        return
    
    filepath = METRIC_PATH + filepath if not filepath.startswith(METRIC_PATH) else filepath
    filepath_obj = Path(filepath)
    filepath_obj.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = logs[0].keys()

    with open(filepath, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(logs)
