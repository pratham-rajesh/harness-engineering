from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def build_model():
    return make_pipeline(StandardScaler(), SVR(C=9.989, epsilon=2.459, gamma=0.006672))
