import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: 0.6161015519568152
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=MultinomialNB(alpha=1.0, fit_prior=False)),
    VarianceThreshold(threshold=0.2),
    StackingEstimator(estimator=MultinomialNB(alpha=1.0, fit_prior=True)),
    StackingEstimator(estimator=GaussianNB()),
    StackingEstimator(estimator=MultinomialNB(alpha=0.1, fit_prior=False)),
    KNeighborsClassifier(n_neighbors=92, p=1, weights="uniform")
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
