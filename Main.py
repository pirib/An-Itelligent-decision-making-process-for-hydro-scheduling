import Data
import Evaluation


from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier as RF
from catboost import CatBoostClassifier

from collections import Counter
import numpy as np

# Models
import RandomForest
import MLPClassifier
import KNN
import SVM
import NaiveBayes
import AdaBoostClassifier
import CatBoost
import XGBoost
import HGB

import matplotlib.pyplot as plt

# Get the data
data = Data.Data(data_num = 19, command_column = 1, 
                 use_pre_process = True, imb_learning = "SMOTE")


#xgb = XGBoost.XGB(data, 0.25)
#print(xgb.accuracy)

# Use Feature Ranking to reduce number of columns
#columns = data.GetFeatureRanking(CatBoostClassifier(verbose=False), 4, False)
#data.KeepColumns(columns)

# rf = RandomForest.RF(data, 0.25, confusion_matrix = True)
# print(rf.accuracy)

# Save the predictions
# predictions = rf.model.predict(data.X_test)
# data.SavePredictions(predictions)


# Testing classifier separately
#cb = CatBoost.CB(data, 0.25, confusion_matrix = True)
#print(cb.accuracy)

#mlp = MLPClassifier.MLP(data, 0.3, confusion_matrix = True)
#print(mlp.accuracy)

#hgb = HGB.HGB(data, 0.25, confusion_matrix = True)
#print(hgb.accuracy)


Evaluation.run_test_batch_ba(RandomForest.RF, data, verbose = True)
#Evaluation.run_test_batch(HGB.HGB, data, verbose = True)




# Permutation importance 
"""
print("\nRandom Forest")
model = RandomForest.RF(data, test_size)
print("Accuracy:", model.accuracy)
Evaluation.perm_importance(model, data)
"""

"""
print("\nSupport Vector Machines")
model = SVM.SVM(data, test_size)
print("Accuracy:", model.accuracy)
Evaluation.perm_importance(model, data)
"""



# Testing grounds















