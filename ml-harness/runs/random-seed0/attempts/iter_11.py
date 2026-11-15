from sklearn.ensemble import HistGradientBoostingRegressor


def build_model():
    return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05055, max_depth=None, l2_regularization=1.17, min_samples_leaf=10, random_state=0)
