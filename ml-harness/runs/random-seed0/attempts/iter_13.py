from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def build_model():
    return make_pipeline(StandardScaler(), Lasso(alpha=0.471, max_iter=20000))
