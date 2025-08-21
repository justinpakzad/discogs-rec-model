import pickle
import re
import argparse
from pathlib import Path
from annoy import AnnoyIndex


DATA_PATH = Path(__file__).parents[1] / "data"

annoy_index = None


def get_n_components():
    with open(DATA_PATH / "n_components.txt", "r") as f:
        n = int(f.read().strip())
    return n


def extract_release_id(url: str) -> str:
    """
    Extract release ID from Discogs URL.

    Args:
        url: Discogs release URL

    Returns:
        int: Extracted release ID
    """
    release_id = int(url.split("release/")[-1].split("-")[0])
    return release_id


def load_mappings():
    """
    Load all pickle mapping files from the mappings directory.

    Returns:
        dict: Dictionary with mapping names as keys and loaded data as values
    """
    mappings = {}
    for item in Path(DATA_PATH).iterdir():
        if not item.name.startswith(".") and item.name.endswith(".pkl"):
            with open(item, "rb") as f:
                mappings[item.name.split(".")[0]] = pickle.load(f)

    return mappings


def load_annoy_index(n: int):
    """
    Load the Annoy index from disk
    """
    global annoy_index
    ann_file_path = str(DATA_PATH / "discogs_rec.ann")
    f = n
    annoy_index = AnnoyIndex(f, "angular")
    annoy_index.load(str(ann_file_path))


def validate_url(url):
    """
    Validate Discogs release URL format.

    Args:
        url: URL to validate

    Returns:
        bool: True if URL matches expected Discogs format, False otherwise
    """
    pattern = r"^https://www\.discogs\.com/release/\d+(?:-[a-zA-Z0-9\-]+)?$"
    return bool(re.match(string=url, pattern=pattern))


def get_nearest_indices(release_id: int, n_recs: int, mappings: dict) -> list[int]:
    """
    Get nearest neighbor indices for a release ID.

    Args:
        release_id: Discogs release ID
        n_recs: Number of recommendations requested

    Returns:
        list[int]: List of nearest neighbor indices, or None if release not found
    """
    item_index = mappings.get("release_id_to_idx").get(release_id)
    if not item_index:
        return None
    nearest_indices = annoy_index.get_nns_by_item(
        item_index, n=n_recs + 25, include_distances=False
    )
    return nearest_indices


def get_n_nearest_recs(url: str, mappings: dict, n_recs: int = 5):
    """
    Extracts release ID from URL, finds similar releases using Annoy index,
    and returns formatted recommendations.

    Args:
        url: Discogs release URL
        n_recs: Number of recommendations to return (default: 5)

    Returns:
        list[dict]: List of recommendations with artist, title, and URL

    """
    valid_url = validate_url(url)
    if not valid_url:
        raise ValueError("Invalid URL")
    release_id = extract_release_id(url)
    indices = get_nearest_indices(
        release_id=release_id, n_recs=n_recs, mappings=mappings
    )

    if not indices:
        raise ValueError("Sorry this release is out of the scope of our model!")
    seen_artists = set()
    recs = []
    for i, idx in enumerate(indices[1:], start=1):

        release_metadata = mappings.get("idx_to_release_info").get(idx)
        artist = release_metadata.get("artist_name")
        title = release_metadata.get("release_title")
        release_id = release_metadata.get("release_id")
        label = release_metadata.get("label_name")
        url = f"https://www.discogs.com/release/{release_id}"
        if artist in seen_artists:
            continue
        recs.append(
            {
                "artist": artist,
                "title": title,
                "label": label,
                "url": url,
            }
        )
        seen_artists.add(artist.strip().lower())
        if i >= n_recs:
            break
    return recs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", help="Enter discogs URL here", type=str)
    args = parser.parse_args()
    mappings = load_mappings()
    n = get_n_components()
    load_annoy_index(n)
    recs = get_n_nearest_recs(
        url=args.url,
        mappings=mappings,
        n_recs=5,
    )
    return recs


if __name__ == "__main__":
    recs = main()
    print(recs)
