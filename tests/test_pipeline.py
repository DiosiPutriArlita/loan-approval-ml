from src.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.generate_data import generate_dataset
from src.preprocessing import (
    build_preprocessor,
    clean_data,
    split_features_target,
    split_train_test,
)


def test_clean_data_removes_duplicates():
    df = generate_dataset(n_samples=200, seed=1)
    assert df.duplicated().sum() > 0
    assert clean_data(df).duplicated().sum() == 0


def test_split_is_stratified_and_sized():
    df = clean_data(generate_dataset(n_samples=500, seed=1))
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    assert len(X_test) == round(len(X) * 0.2)
    assert abs(y_train.mean() - y_test.mean()) < 0.02


def test_preprocessor_fills_missing_values():
    df = clean_data(generate_dataset(n_samples=300, seed=2))
    X, _ = split_features_target(df)
    assert X.isna().sum().sum() > 0
    transformed = build_preprocessor().fit_transform(X)
    dense = transformed.toarray() if hasattr(transformed, "toarray") else transformed
    assert not (dense != dense).any()


def test_feature_lists_exist_in_dataset():
    df = generate_dataset(n_samples=50, seed=3)
    for column in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
        assert column in df.columns
