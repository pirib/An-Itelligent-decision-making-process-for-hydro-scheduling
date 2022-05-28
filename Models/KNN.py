import Evaluation

from sklearn.neighbors import KNeighborsClassifier


class KNN:
    
    name = "K-nearest Neighbours"
    
    def __init__(self, data, test_size, confusion_matrix = False, verbose = False):

        # Divide the set into training and testing
        self.X_train, self.X_test, self.Y_train, self.Y_test = data.Split(test_size = test_size)
        
        # Print some pre-training stats
        if verbose:
            print("Model: " + self.name)
            Evaluation.print_stats(data.X.shape[0], test_size)
        
        # Instantiate and fit the model
        self.model = KNeighborsClassifier(n_neighbors = 15)
        self.model = self.model.fit(self.X_train, self.Y_train)
        
        # Evaluate the prediction
        self.accuracy = Evaluation.evaluate(self.model, self.X_test, self.Y_test, confusion_matrix , verbose)