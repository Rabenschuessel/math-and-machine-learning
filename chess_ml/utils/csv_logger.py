import csv
from pathlib import Path
from typing import List, Dict, Any


class CSVLogger:
    """
    Simple CSV logger for structured experiment metrics.

    This logger appends one row per call to `log()` and is intended
    for batch-level metrics in reinforcement learning experiments.

    Typical use cases:
        - training loss per batch
        - average rewards per component
        - win / draw / loss statistics
        - episode length statistics

    The resulting CSV file can easily be loaded into pandas, Excel,
    or plotting tools for later analysis.
    """

    def __init__(self, path: Path, fieldnames: List[str]):
        """
        Initialize the CSV logger.

        Parameters
        ----------
        path : Path
            Path to the CSV file (e.g. logs/rl/experiment-X/metrics.csv).
        fieldnames : list of str
            Column names of the CSV file. Must match the keys
            passed to `log()`.
        """
        self.path = path
        self.fieldnames = fieldnames

        # Ensure that the directory exists
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # Create file and write header if it does not exist yet
        if not self.path.exists():
            with open(self.path, mode="w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log(self, row: Dict[str, Any]):
        """
        Append a single row of metrics to the CSV file.

        Parameters
        ----------
        row : dict
            Dictionary mapping column names (fieldnames) to values.
            All fieldnames must be present in the dictionary.
        """

        # Optional safety check: ensure all fields are present
        missing = set(self.fieldnames) - set(row.keys())
        if missing:
            raise ValueError(
                f"Missing fields in CSV log row: {missing}"
            )

        with open(self.path, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)
