import gdown
import ast
from pathlib import Path
import pickle
import re
import pandas as pd
import numpy as np
from annoy import AnnoyIndex


def download_discogs_dataset() -> None:
    """
    Download the Discogs dataset from Google Drive to the local data directory.
    """
    path = Path.cwd().parents[0] / "data" / "training_data"
    path.mkdir(parents=True, exist_ok=True)
    url = "https://drive.google.com/uc?export=download&id=1TAdQU8xTK-TiIk3_wGVndHkbokUsSgNS"
    gdown.download(url, str(f"{path}/discogs_dataset.parquet"))


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
        - idx_to_release_id: Maps DataFrame indices to release IDs
        - idx_to_title: Maps DataFrame indices to release titles
        - idx_to_artist: Maps DataFrame indices to formatted artist names
    """
    df["artist_name"] = (
        df["artist_name"]
        .astype(str)
        .apply(lambda x: ast.literal_eval(re.sub(r"'\s+'", "', '", x)))
    )
    # Build mappings for displaying artist/release on web app
    release_id_to_idx = {
        release_id: idx for idx, release_id in enumerate(df["release_id"])
    }
    idx_to_release_id = {
        idx: release_id for release_id, idx in release_id_to_idx.items()
    }
    release_id_to_title = {
        idx: release_title for idx, release_title in enumerate(df["release_title"])
    }
    release_id_to_artist = {
        idx: " / ".join(artist) for idx, artist in enumerate(df["artist_name"])
    }
    mappings = {
        "release_id_to_idx": release_id_to_idx,
        "idx_to_release_id": idx_to_release_id,
        "idx_to_title": release_id_to_title,
        "idx_to_artist": release_id_to_artist,
    }
    return mappings


def write_mappings(mappings: dict[str, dict[int | str, int | str]]) -> None:
    """
    Write mapping dictionaries to pickle files in the data/mappings directory.
    Args:
        mappings: Dictionary containing mapping dictionaries to serialize
    """
    # Mounted path
    dirpath = Path("/data/mappings")
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
    ann_dir = Path("/data/ann_files")
    ann_dir.mkdir(parents=True, exist_ok=True)

    # Creating annoy index
    t = AnnoyIndex(f, "angular")
    for i in range(matrix.shape[0]):
        t.add_item(i, matrix[i])
    t.build(n_trees)
    t.save(f"/data/ann_files/{file_name}")
