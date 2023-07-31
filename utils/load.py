import pathlib

import pandas as pd


def load_training_data(features_csv: str, labels_csv: str, images_root: str) -> tuple[pd.DataFrame, pd.DataFrame]:

    features = pd.read_csv(features_csv, index_col="id")
    features['filepath'] = features['filepath'].apply(lambda path: pathlib.Path(images_root) / str(path)) # type: ignore
    train_labels = pd.read_csv(labels_csv, index_col="id")

    y = train_labels
    x = features.loc[y.index].filepath.to_frame()

    return x, y
