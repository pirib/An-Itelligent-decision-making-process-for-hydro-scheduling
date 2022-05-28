import Evaluation

from catboost import CatBoostClassifier


class CB:
    
    name = "CatBoost"
    
    def __init__(self, data, test_size, confusion_matrix = False, verbose = False):
    
        # Divide the set into training and testing
        self.X_train, self.X_test, self.Y_train, self.Y_test = data.Split(test_size = test_size)

        # Print some pre-training stats
        if verbose:
            print("Model: " + self.name)
            Evaluation.print_stats(data.X.shape[0], test_size)

        # Instantiate and fit the model
        self.model = CatBoostClassifier(iterations=150,
                                   objective="CrossEntropy",
                                   depth=4,
                                   random_strength=18,
                                   bagging_temperature=0.688,
                                   boosting_type='Ordered',
                                   learning_rate=0.0877,
                                   loss_function='Logloss',
                                   verbose = verbose)
        
        self.model.fit(self.X_train, self.Y_train)
        
        # make the prediction using the resulting model
        self.accuracy = Evaluation.evaluate(self.model, self.X_test, self.Y_test, confusion_matrix , verbose)




