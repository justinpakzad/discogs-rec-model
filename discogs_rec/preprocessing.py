from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.decomposition import TruncatedSVD
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix as CSRMatrix


weights_dict = {
    "want_to_have_ratio": 0.4,
    "have": 0.4,
    "want": 0.4,
    "avg_rating": 0.4,
    "low": 0.4,
    "median": 0.4,
    "high": 0.4,
    "ratings": 0.1,
    "video_count": 0.1,
    "n_styles": 0.4,
    "country": 0.7,
    "release_year": 1.0,
    "styles": 1.0,
}


def scale_features(df: pd.DataFrame, weights_dict: dict) -> dict[str, np.ndarray]:
    """
    Scale numerical features using StandardScaler and apply feature weights.

    Args:
        df: DataFrame containing the features to scale

    Returns:
        Dictionary mapping feature names to scaled and weighted feature arrays
    """
    scaler = StandardScaler()
    scaled_features = {
        feature: scaler.fit_transform(df[[feature]]) * weights_dict.get(feature, 1)
        for feature in weights_dict.keys()
        if feature not in ["country", "release_year", "styles"]
    }
    return scaled_features


def one_hot_encode_features(df: pd.DataFrame) -> dict[str, CSRMatrix]:
    """
    One-hot encode categorical features and apply feature weights.

    Args:
        df: DataFrame containing country and release_year columns

    Returns:
        Dictionary mapping feature names to weighted sparse matrices
    """
    ohe = OneHotEncoder()
    encoded_features = {
        "countries": csr_matrix(ohe.fit_transform(df[["country"]]))
        * weights_dict.get("country", 1),
        "year": csr_matrix(ohe.fit_transform(df[["release_year"]]))
        * weights_dict.get("release_year", 1),
    }
    return encoded_features


def ml_encode_features(df: pd.DataFrame) -> dict[str, CSRMatrix]:
    """
    Multi-label binarize the styles feature and apply feature weights.

    Args:
        df: DataFrame containing a 'styles' column with lists of style labels

    Returns:
        Dictionary mapping 'styles' to weighted sparse matrix
    """
    mlb = MultiLabelBinarizer()
    encoded_features = {
        "styles": csr_matrix(
            mlb.fit_transform(df["styles"]) * weights_dict.get("styles", 1)
        )
    }
    return encoded_features


def fill_nulls(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Fill null values in specified columns with zeros using SimpleImputer.

    Args:
        df: DataFrame to impute missing values in
        columns: List of column names to fill nulls for

    Returns:
        DataFrame with null values filled
    """
    imputer = SimpleImputer(fill_value=0, strategy="constant")
    df[columns] = imputer.fit_transform(df[columns])
    return df


def process_all_features(
    df: pd.DataFrame,
    columns: list[str],
    weights_dict: dict,
    features: list[str] | None = None,
) -> CSRMatrix:
    """
    Process all features through scaling, encoding, and stacking operations.

    Args:
        df: DataFrame containing all features to process
        columns: List of column names to fill nulls for
        features: Optional list of specific features to select. If None, uses all features.

    Returns:
        Horizontally stacked sparse matrix of all processed features
    """
    imputed_df = fill_nulls(df, columns)
    scaled_features = scale_features(imputed_df, weights_dict)
    ohe_features = one_hot_encode_features(imputed_df)
    multi_label_features = ml_encode_features(imputed_df)
    features_dict = {**scaled_features, **ohe_features, **multi_label_features}
    selected_features = (
        {k: v for k, v in features_dict.items() if k in features}
        if features
        else features_dict
    )
    features_stacked = hstack([val for val in selected_features.values()])
    return features_stacked


def reduce_dimensionality(
    feature_matrix: np.ndarray | CSRMatrix, n_components: int = 200
) -> np.ndarray:
    """
    Reduce dimensionality of feature matrix using TruncatedSVD.

    Args:
        feature_matrix: Input feature matrix (dense or sparse)
        n_components: Target number of components. Will be capped at matrix width - 1.

    Returns:
        Reduced feature matrix with n_components dimensions
    """
    # Make sure default n_components does not exceed the number of features
    # if it does, use the feature matrix shape
    max_components = min(n_components, feature_matrix.shape[1] - 1)
    svd = TruncatedSVD(n_components=max_components)
    reduced_features = svd.fit_transform(feature_matrix)
    return reduced_features


def write_n_components(n_components: int) -> None:
    """
    Write the number of components to a text file for dynamic Annoy parameter configuration.

    Args:
        n_components: Number of components to write to config file
    """
    # Write the n_components to a txt file
    # so that the annoy f parameter can be dynamically updated instead of a fixed 150
    config_path = Path("/data")
    config_path.mkdir(parents=True, exist_ok=True)
    with open(f"{config_path}/n_components.txt", "w") as f:
        f.write(str(n_components))
