import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import PolynomialFeatures
from tpot.builtins import OneHotEncoder, StackingEstimator

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: 0.978189124487004
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=SGDClassifier(alpha=0.001, eta0=1.0, fit_intercept=False, l1_ratio=0.75, learning_rate="constant", loss="perceptron", penalty="elasticnet", power_t=0.5)),
    PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
    OneHotEncoder(minimum_fraction=0.1, sparse=False, threshold=10),
    OneHotEncoder(minimum_fraction=0.25, sparse=False, threshold=10),
    PCA(iterated_power=7, svd_solver="randomized"),
    GradientBoostingClassifier(learning_rate=0.5, max_depth=9, max_features=0.05, min_samples_leaf=13, min_samples_split=9, n_estimators=100, subsample=0.9500000000000001)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
