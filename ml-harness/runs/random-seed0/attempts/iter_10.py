from sklearn.ensemble import RandomForestRegressor


def build_model():
    return RandomForestRegressor(n_estimators=200, max_depth=6, min_samples_leaf=5, max_features=0.3, random_state=0, n_jobs=1)
