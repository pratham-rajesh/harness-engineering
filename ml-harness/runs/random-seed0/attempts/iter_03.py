from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def build_model():
    return make_pipeline(StandardScaler(), PolynomialFeatures(degree=2, interaction_only=False, include_bias=False),
                         StandardScaler(), Ridge(alpha=86.04))
