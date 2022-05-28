import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import StandardScaler
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: 0.765702614379085
exported_pipeline = make_pipeline(
    make_union(
        make_pipeline(
            StandardScaler(),
            FastICA(tol=0.8500000000000001)
        ),
        RFE(estimator=ExtraTreesClassifier(criterion="entropy", max_features=0.6000000000000001, n_estimators=100), step=0.8)
    ),
    ExtraTreesClassifier(bootstrap=False, criterion="gini", max_features=0.3, min_samples_leaf=1, min_samples_split=3, n_estimators=100)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
