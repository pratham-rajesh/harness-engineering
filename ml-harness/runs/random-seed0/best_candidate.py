from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def build_model():
    base = [('ridge', make_pipeline(StandardScaler(), Ridge(alpha=0.8342))),
            ('svr', make_pipeline(StandardScaler(), SVR(C=50, epsilon=10, gamma=0.01))),
            ('gbr', GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=2, subsample=0.7, random_state=0))]
    return StackingRegressor(estimators=base, final_estimator=Ridge(alpha=1.0), cv=3)
