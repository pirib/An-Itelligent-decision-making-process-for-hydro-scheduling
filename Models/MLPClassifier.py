import Evaluation

from sklearn.neural_network import MLPClassifier


class MLP:
    
    name = "Multi Layer Perceptron"
    
    def __init__(self, data, test_size = 0.25, confusion_matrix = False, verbose = False):
        
        # Divide the set into training and testing
        self.X_train, self.X_test, self.Y_train, self.Y_test = data.Split(test_size = test_size)
        
        # Print some pre-training stats
        if verbose:
            print("Model: Multi Layer Perceptron Classifier")
            Evaluation.print_stats(data.X.shape[0], test_size)
        
        # Instantiate and fit the model
        self.model = MLPClassifier(activation = 'relu', 
                                   hidden_layer_sizes=[32], 
                                   solver = 'adam', 
                                   alpha = 1e-3, 
                                   learning_rate_init = 0.001,
                                   batch_size=1)
        
        self.model = self.model.fit(self.X_train, self.Y_train)
        
        # Evaluate the prediction
        self.accuracy = Evaluation.evaluate(self.model, self.X_test, self.Y_test, confusion_matrix , verbose)
        
