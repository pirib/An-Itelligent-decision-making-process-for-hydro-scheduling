"""
Sanity check for the predictor classes using toy datasets
"""

import MLPClassifier
import RandomForest
import HGB

import ToyData



data = ToyData.Data()

mlp = MLPClassifier.MLP(data, 0.25, confusion_matrix = True)
rf = RandomForest.RF(data, 0.25, confusion_matrix = True)
ngb = HGB.HGB(data, 0.25, confusion_matrix = True)