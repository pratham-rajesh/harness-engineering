from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def build_model():
    return make_pipeline(StandardScaler(), Ridge(alpha=2.559))
