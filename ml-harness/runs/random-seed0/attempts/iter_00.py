"""Baseline: predict the training mean. Anything worth keeping must beat this by a lot."""
from sklearn.dummy import DummyRegressor


def build_model():
    return DummyRegressor(strategy="mean")
