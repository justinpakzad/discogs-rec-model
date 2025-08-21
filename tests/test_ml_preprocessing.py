import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.sparse import csr_matrix

from discogs_rec.preprocessing import (
    scale_features,
    one_hot_encode_features,
    ml_encode_features,
    fill_nulls,
)


@pytest.fixture
def weights_dict():
    return {
        "median": 0.4,
        "ratings": 0.1,
        "have": 0.4,
        "country": 0.5,
        "release_year": 0.9,
    }


def test_one_hot_encode_features():
    df = pd.DataFrame(
        {"country": ["DE", "UK", "US", "UK"], "release_year": [2020, 2021, 2020, 2022]}
    )

    encoded = one_hot_encode_features(df)

    ohe = OneHotEncoder()
    expected_countries = csr_matrix(ohe.fit_transform(df[["country"]])) * 0.7

    np.testing.assert_array_equal(
        encoded["countries"].toarray(), expected_countries.toarray()
    )

    expected_years = csr_matrix(ohe.fit_transform(df[["release_year"]])) * 1.0
    np.testing.assert_array_equal(encoded["year"].toarray(), expected_years.toarray())


def test_fill_nulls():
    df = pd.DataFrame({"ratings": [10.0, None, None]})
    columns = ["ratings"]

    result = fill_nulls(df, columns)
    expected = pd.DataFrame({"ratings": [10.0, 0.0, 0.0]})

    pd.testing.assert_frame_equal(result, expected)


def test_ml_encode_features():
    df = pd.DataFrame(
        {"styles": [["techno", "house", "electro"], ["tech-house", "techno"]]}
    )

    encoded = ml_encode_features(df)

    mlb = MultiLabelBinarizer()
    expected = csr_matrix(mlb.fit_transform(df["styles"]) * 1.0)

    np.testing.assert_array_equal(encoded["styles"].toarray(), expected.toarray())


def test_scale_features(weights_dict):
    df = pd.DataFrame(
        {
            "median": [12.99, 25.0, 58.99],
            "ratings": [100, 200, 300],
            "have": [5, 10, 15],
        }
    )

    scaled = scale_features(df, weights_dict)

    scaler = StandardScaler()
    expected_median = scaler.fit_transform(df[["median"]]) * 0.4
    np.testing.assert_array_equal(scaled["median"], expected_median)

    expected_ratings = scaler.fit_transform(df[["ratings"]]) * 0.1
    np.testing.assert_array_equal(scaled["ratings"], expected_ratings)

    expected_have = scaler.fit_transform(df[["have"]]) * 0.4
    np.testing.assert_array_equal(scaled["have"], expected_have)


def test_scale_features_only_includes_correct_features(weights_dict):
    df = pd.DataFrame(
        {
            "median": [12.99, 25.0, 58.99],
            "country": ["US", "UK", "DE"],
            "styles": [["techno"], ["house"], ["electro"]],
            "ratings": [100, 200, 300],
            "have": [200, 100, 10],
        }
    )

    scaled = scale_features(df, weights_dict)

    assert "country" not in scaled
    assert "styles" not in scaled
    assert "median" in scaled
    assert "ratings" in scaled
