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


def _find_all_dataset_files() -> List[Path]:
    """
    Find all Wyckoff dataset files in the data directory.
    Returns a list of paths to all matching CSV files.
    """
    dataset_files = []
    
    # Known dataset patterns
    known_patterns = [
        "WyckoffDataset-*.csv",
        "WycoffDataset-*.csv",
    ]
    
    for pattern in known_patterns:
        dataset_files.extend(DATA_DIR.glob(pattern))
    
    # Remove duplicates while preserving order
    seen = set()
    unique_files = []
    for f in dataset_files:
        if f not in seen:
            seen.add(f)
            unique_files.append(f)
    
    return unique_files


def load_wyckoff_dataset(relative_path: Optional[str] = None, load_all: bool = True) -> pd.DataFrame:
    """
    Load the Wyckoff Q&A dataset as a DataFrame.

    Parameters:
    - relative_path: Optional specific file to load
    - load_all: If True, loads and combines all WyckoffDataset-*.csv files
                If False, loads only the first/specified file

    - Auto-detects the CSV file in /data (with support for Wycoff/Wyckoff spelling)
    - Normalizes column names:
        Questions -> Questions
        Answers   -> Answers
        Label     -> Label
    - Returns a clean DataFrame ready for RAG.
    """
    
    if relative_path is not None:
        # Load specific file
        csv_path = _find_dataset_file(relative_path)
        df = _load_single_dataset(csv_path)
    elif load_all:
        # Load and combine all dataset files
        dataset_files = _find_all_dataset_files()
        
        if not dataset_files:
            # Fallback to single file detection
            csv_path = _find_dataset_file(None)
            df = _load_single_dataset(csv_path)
        else:
            # Load and concatenate all datasets
            dataframes = []
            for csv_path in dataset_files:
                try:
                    single_df = _load_single_dataset(csv_path)
                    # Add source column to track which dataset each row came from
                    single_df["_source"] = csv_path.name
                    dataframes.append(single_df)
                except Exception as e:
                    pass  # Silently skip files that can't be loaded
            
            if not dataframes:
                raise FileNotFoundError(
                    f"No valid CSV datasets found in {DATA_DIR}."
                )
            
            # Combine all dataframes
            df = pd.concat(dataframes, ignore_index=True)
            
            # Remove duplicates based on Questions column
            original_len = len(df)
            df = df.drop_duplicates(subset=["Questions"], keep="first")
            
            # Drop the source column before returning (optional - keep if you want tracking)
            df = df.drop(columns=["_source"], errors="ignore")
    else:
        # Load single file (original behavior)
        csv_path = _find_dataset_file(None)
        df = _load_single_dataset(csv_path)
    
    return df[["Questions", "Answers", "Label"]].copy()


def _load_single_dataset(csv_path: Path) -> pd.DataFrame:
    """
    Load and normalize a single dataset file.
    """
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


def get_dataset_info() -> dict:
    """
    Get information about loaded datasets.
    Useful for debugging and UI display.
    """
    dataset_files = _find_all_dataset_files()
    info = {
        "data_directory": str(DATA_DIR),
        "datasets_found": len(dataset_files),
        "dataset_files": [f.name for f in dataset_files],
        "details": []
    }
    
    for csv_path in dataset_files:
        try:
            df = pd.read_csv(csv_path)
            info["details"].append({
                "name": csv_path.name,
                "rows": len(df),
                "columns": list(df.columns)
            })
        except Exception as e:
            info["details"].append({
                "name": csv_path.name,
                "error": str(e)
            })
    
    return info