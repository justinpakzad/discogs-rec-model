import shutil
import ast
import math
import pickle
import re
import pandas as pd
import numpy as np
from annoy import AnnoyIndex
from pathlib import Path
from huggingface_hub import hf_hub_download


def download_discogs_dataset() -> None:
    """
    Download the Discogs dataset from Hugging Face to the local data directory.
    """
    path = Path("/data")
    path.mkdir(parents=True, exist_ok=True)
    for item in path.iterdir():
        if item.is_file() and item.stem == "discogs_dataset":
            return
    hf_hub_download(
        repo_id="justinp303/discogs-recommender-model",
        repo_type="dataset",
        filename="discogs_dataset.parquet",
        local_dir=str(path),
    )
    shutil.rmtree(path / ".cache")


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess DataFrame by removing duplicates and standardizing values.

    Args:
        df: DataFrame containing release data to clean

    Returns:
        Cleaned DataFrame with duplicates removed and standardized values
    """
    df = df.drop_duplicates(
        subset=["release_title", "label_name", "release_year", "catno"], keep="first"
    )
    df["n_styles"] = df["styles"].apply(len)

    df["want_to_have_ratio"] = df["want_to_have_ratio"].round(3)

    df.loc[df["catno"] == "none", "catno"] = None
    return df


def clean_mappings(records: list[dict]) -> list[dict]:
    """
    Clean NaN values from a list of record dictionaries.

    Args:
        records: List of dictionaries containing record data with potential NaN values

    Returns:
        List of dictionaries with NaN values replaced by None
    """
    cleaned_records = []
    for record in records:
        cleaned_record = {}
        for key, value in record.items():
            if isinstance(value, float) and math.isnan(value):
                cleaned_record[key] = None
            else:
                cleaned_record[key] = value
        cleaned_records.append(cleaned_record)
    return cleaned_records


def create_mappings(
    df: pd.DataFrame,
) -> dict[str, dict[int | str, int | str]]:
    """
    Create mapping dictionaries for release IDs, titles, and artists.
    Args:
        df: DataFrame containing release_id, release_title, and artist_name columns
    Returns:
        Dictionary containing four mapping dictionaries:
        - release_id_to_idx: Maps release IDs to DataFrame indices
        - idx_to_release_info: Maps DataFrame indices to release IDs

    """
    df["artist_name"] = (
        df["artist_name"]
        .astype(str)
        .apply(lambda x: ast.literal_eval(re.sub(r"'\s+'", "', '", x)))
    )

    df["artist_name"] = df["artist_name"].apply(lambda x: " / ".join(x))

    df["styles"] = df["styles"].apply(list)
    # Build mappings for displaying artist/release on web app
    release_id_to_idx = {
        release_id: idx for idx, release_id in enumerate(df["release_id"])
    }
    columns = [
        "release_id",
        "artist_name",
        "styles",
        "release_title",
        "country",
        "catno",
        "label_name",
        "release_year",
        "want",
        "have",
        "want_to_have_ratio",
        "video_count",
        "low",
        "median",
        "high",
    ]

    records = df[columns].to_dict("records")
    cleaned_records = clean_mappings(records)
    idx_to_release_info = {idx: record for idx, record in enumerate(cleaned_records)}
    mappings = {
        "release_id_to_idx": release_id_to_idx,
        "idx_to_release_info": idx_to_release_info,
    }
    return mappings


def write_mappings(mappings: dict[str, dict[int | str, int | str]]) -> None:
    """
    Write mapping dictionaries to pickle files in the data/mappings directory.
    Args:
        mappings: Dictionary containing mapping dictionaries to serialize
    """
    # Mounted path
    dirpath = Path("/data")
    dirpath.mkdir(parents=True, exist_ok=True)
    for key, value in mappings.items():
        file_dest = dirpath / f"{key}.pkl"
        with open(file_dest, "wb") as fp:
            pickle.dump(value, fp)


def build_annoy_index(
    matrix: np.ndarray, file_name: str, f: int = 150, n_trees: int = 250
) -> None:
    """
    Build and save an Annoy index for approximate nearest neighbor search.

    Args:
        matrix: Feature matrix where each row represents an item
        file_name: Name of the file to save the Annoy index
        f: Number of dimensions in the feature vectors
        n_trees: Number of trees to build (more trees = better accuracy, slower build)
    """
    # Create Annoy index directory if it doesn't exist
    ann_dir = Path("/data")
    ann_dir.mkdir(parents=True, exist_ok=True)

    # Creating annoy index
    t = AnnoyIndex(f, "angular")
    for i in range(matrix.shape[0]):
        t.add_item(i, matrix[i])
    t.build(n_trees)
    t.save(f"/data/{file_name}")
