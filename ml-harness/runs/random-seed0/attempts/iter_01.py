from sklearn.linear_model import ElasticNet
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def build_model():
    return make_pipeline(StandardScaler(), ElasticNet(alpha=0.9298, l1_ratio=0.7, max_iter=20000))
