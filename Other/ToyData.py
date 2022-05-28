"""
Toy datasets for testing on the custom wrappers
"""

from sklearn.datasets import load_iris, load_wine, load_breast_cancer

from sklearn.model_selection import train_test_split



class Data:
    
    def __init__(self, dataset= 'iris'):
        
        if dataset == 'iris':
            self.X, self.Y = load_iris(return_X_y=True) 
        elif dataset == 'wine':
            self.X, self.Y = load_wine(return_X_y=True) 
        elif dataset == 'cancer':
            self.X, self.Y = load_wine(return_X_y=True) 
        else:
            print("No such dummy dataset found: " + dataset)
            raise

    def Split(self, test_size = 0.25, stratify_y = True):
        
        st = self.Y if stratify_y else None
        
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.Y, stratify = st, test_size = test_size, random_state = 0 )
        
        return X_train, X_test, y_train, y_test
