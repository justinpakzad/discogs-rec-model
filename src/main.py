import ast
import pandas as pd
import pickle
from pathlib import Path
from annoy import AnnoyIndex
from preprocessor import BuildFeatures
import argparse


def approx_nearest_neighbor(matrix, file_name, f=150, n_trees=500):
    # creating annoy index
    t = AnnoyIndex(f, "angular")
    for i in range(matrix.shape[0]):
        t.add_item(i, matrix[i])
    t.build(n_trees)
    t.save(f"/ann_files/{file_name}")


def create_mappings(df):
    df["artist_name"] = df["artist_name"].apply(lambda x: ast.literal_eval(x))
    # build mappings for displaying artist/release on web app
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
        idx: ", ".join(artist) for idx, artist in enumerate(df["artist_name"])
    }
    mappings = {
        "release_id_to_idx": release_id_to_idx,
        "idx_to_release_id": idx_to_release_id,
        "idx_to_title": release_id_to_title,
        "idx_to_artist": release_id_to_artist,
    }
    return mappings


def write_mappings(mappings):
    # mounted path
    dirpath = Path("/mappings") 
    dirpath.mkdir(exist_ok=True)
    for key, value in mappings.items():
        file_dest = dirpath / f"{key}.pkl"
        with open(file_dest, "wb") as fp:
            pickle.dump(value, fp)


def arg_parse():
    parser = argparse.ArgumentParser(
        description="Process feature selection for data transformation."
    )
    parser.add_argument(
        "--features",
        nargs="+",
        type=str,
        help="List of features to include in the processing",
    )
    args = parser.parse_args()
    return args


def main():
    data_path = Path("/data")  # mounted
    df = pd.read_csv(f"{data_path}/discogs_rec_dataset.csv")
    df_cleaned = df.drop_duplicates(
        subset=["release_title", "label_name", "release_year", "catno"], keep="first"
    )
    args = arg_parse()
    feature_builder = BuildFeatures(features=args.features)
    feature_matrix = feature_builder.process_features(df_cleaned, args.features)
    reduced_features = feature_builder.reduce_dimensionality(feature_matrix)
    mappings = create_mappings(df_cleaned)
    write_mappings(mappings)
    approx_nearest_neighbor(reduced_features, file_name="discogs_rec.ann")


if __name__ == "__main__":
    main()
