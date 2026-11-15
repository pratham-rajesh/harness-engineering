from sklearn.ensemble import GradientBoostingRegressor


def build_model():
    return GradientBoostingRegressor(n_estimators=100, learning_rate=0.01016, max_depth=1, subsample=0.7, random_state=0)
