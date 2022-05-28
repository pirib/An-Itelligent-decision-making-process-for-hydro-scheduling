import xgboost as xgb

import numpy as np
import sklearn.metrics

import Evaluation


class XGB:
        
    name = "XGBoost"
        
    def __init__(self, data, test_size, confusion_matrix = False, verbose = False):
    
        # Divide the set into training and testing
        self.X_train, self.X_test, self.Y_train, self.Y_test = data.Split(test_size = test_size)
        
        if verbose:
            print(name)
            Evaluation.print_stats(data.X.shape[0], test_size)
            
        # Get the model
        param_dist = {'objective':'binary:logistic', 'n_estimators': 200}
        model = xgb.XGBClassifier(**param_dist)
        
        # Fit the model
        model.fit(self.X_train, self.Y_train,   
                  eval_set=[(self.X_train, self.Y_train), (self.X_test, self.Y_test)], eval_metric='auc', 
                  early_stopping_rounds = 10,
                  verbose=True)
        
        preds = model.predict(self.X_test)
        
        pred_labels = np.rint(preds)
        self.accuracy = sklearn.metrics.accuracy_score(self.Y_test, pred_labels)
        