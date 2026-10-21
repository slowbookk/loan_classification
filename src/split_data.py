from sklearn.model_selection import train_test_split

from src.config import RANDOM_STATE, TEST_SIZE


def separate_target(data):
    X = data.drop(columns="target")
    y = data["target"].astype(int)
    return X, y


def train_test(X, y):
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )