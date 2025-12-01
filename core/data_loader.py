# core/data_loader.py
import pandas as pd
from pathlib import Path
from typing import Optional, List


# Project root = wyckoff-trading-platform/
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def _find_dataset_file(explicit_relative: Optional[str] = None) -> Path:
    """
    Resolve the dataset path safely.

    Priority:
    1) If explicit_relative is provided and exists, use it.
    2) Try known filenames in /data.
    3) Fall back to the first CSV in /data.
    """
    # 1) If caller passed a path
    if explicit_relative is not None:
        candidate = BASE_DIR / explicit_relative
        if candidate.exists():
            return candidate
        # Also allow direct path relative to data/
        candidate2 = DATA_DIR / explicit_relative
        if candidate2.exists():
            return candidate2
        raise FileNotFoundError(f"Dataset not found at: {candidate}")

    # 2) Known filenames (handle both spellings)
    known_names: List[str] = [
        "WyckoffDataset-1.csv",
        "WycoffDataset-1.csv",  # your current file name
    ]
    for name in known_names:
        candidate = DATA_DIR / name
        if candidate.exists():
            return candidate

    # 3) Any CSV in /data
    csv_files = list(DATA_DIR.glob("*.csv"))
    if csv_files:
        return csv_files[0]

    raise FileNotFoundError(
        f"No CSV dataset found in {DATA_DIR}. "
        f"Expected something like 'WyckoffDataset-1.csv'."
    )


def load_wyckoff_dataset(relative_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load the Wyckoff Q&A dataset as a DataFrame.

    - Auto-detects the CSV file in /data (with support for Wycoff/Wyckoff spelling)
    - Normalizes column names:
        Questions -> Questions
        Answers   -> Answers
        Label     -> Label
    - Returns a clean DataFrame ready for RAG.
    """
    csv_path = _find_dataset_file(relative_path)

    df = pd.read_csv(csv_path)

    # Normalize column names (strip spaces, unify case)
    df.columns = [c.strip() for c in df.columns]

    # Expected original headers in your CSV
    col_map = {}
    for col in df.columns:
        low = col.lower()
        if low in ("questions", "question"):
            col_map[col] = "Questions"
        elif low in ("answers", "answer"):
            col_map[col] = "Answers"
        elif low.strip() == "label":
            col_map[col] = "Label"

    df = df.rename(columns=col_map)

    required = {"Questions", "Answers", "Label"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"Dataset must contain columns {required}, "
            f"but found {df.columns.tolist()} in {csv_path}"
        )

    return df[["Questions", "Answers", "Label"]].copy()
