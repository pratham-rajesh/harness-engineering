from sklearn.ensemble import GradientBoostingRegressor


def build_model():
    return GradientBoostingRegressor(n_estimators=50, learning_rate=0.03069, max_depth=2, subsample=0.7, random_state=0)
