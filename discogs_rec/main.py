import argparse
import logging
import pandas as pd
from pathlib import Path
from discogs_rec.preprocessing import (
    process_all_features,
    reduce_dimensionality,
    write_n_components,
    weights_dict,
)
from discogs_rec.utils import (
    download_discogs_dataset,
    create_mappings,
    write_mappings,
    build_annoy_index,
    clean_df,
)


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("discogs_rec")


def arg_parse() -> argparse.Namespace:
    """
    Parse command line arguments for feature selection.

    Returns:
        Parsed arguments containing optional list of features to process
    """
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


def main() -> None:
    logger.info("Downloading dataset...")
    download_discogs_dataset()

    data_path = Path("/data")  # mounted
    df = pd.read_parquet(f"{data_path}/discogs_dataset.parquet")
    df = df.drop(
        columns=["is_master_release", "name_variations", "real_name", "format"]
    )
    # clean duplicates and add derived features
    logger.info("Cleaning and transforming data..")
    df_cleaned = clean_df(df)
    args = arg_parse()

    cols_to_impute = [
        "have",
        "want",
        "avg_rating",
        "ratings",
        "low",
        "median",
        "high",
        "want_to_have_ratio",
        "video_count",
    ]

    # process features
    feature_matrix = process_all_features(
        df=df_cleaned,
        columns=cols_to_impute,
        features=args.features,
        weights_dict=weights_dict,
    )

    # reduce dimensionality
    reduced_features = reduce_dimensionality(feature_matrix=feature_matrix)
    n_components = reduced_features.shape[1]
    write_n_components(n_components=n_components)

    # create and save mappings
    logger.info("Writing mappings..")
    mappings = create_mappings(df_cleaned)
    write_mappings(mappings)

    # build similarity index
    logger.info("Building Annoy index ..")
    build_annoy_index(reduced_features, f=n_components, file_name="discogs_rec.ann")

    logger.info("Pipeline completed")


if __name__ == "__main__":
    main()
