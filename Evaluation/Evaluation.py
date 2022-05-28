from collections import Counter

import numpy as np

from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, balanced_accuracy_score


# Namespace to perform evaluations on the trained models

# Evaluates the TPOT using balanced accuracy 
def evaluate_TPOT(predictions, Y_test, print_confusion_matrix = True, verbose = True):

    errors = np.count_nonzero(predictions != Y_test)
    
    balanced_accuracy = balanced_accuracy_score(Y_test, predictions)
    
    if verbose:    
        # Print some evaluation info
        print('Model Performance')
        print('errors=%d, balanced_accuracy=%.2f%%' % (errors, balanced_accuracy * 100))
    
        print("True Labels:")
        print(np.asarray(Y_test))
        print("Predictions:")
        print(predictions)

    if print_confusion_matrix:
        # If it is a binary classification
        if len(Counter(Y_test)) == 2:
    
            tn, fp, fn, tp = confusion_matrix(Y_test, predictions).ravel()        
            print("Confusion matrix")
            print("Correctly predicted 0: %d" % (tn))
            print("Correctly predicted 1: %d" % (tp))
            print("Incorrectly predicted 0: %d" % (fn))
            print("Incorrectly predicted 1: %d" % (fp))
            print("Accuracy for label 0: %d" % ((tn*100)/(tn+fn)))
            print("Accuracy for label 1: %d" % ((tp*100)/(tp+fp)))
        
        else:

            print("Confusion matrix")
            print("Horizontally - Predicted labels")
            print("Vertically - True labels")
            labels = list(range(len(Counter(Y_test))))
            cm = confusion_matrix(Y_test, predictions )

            print("      \t", end = '')
            for i in labels:
                print("%5d \t " % i, end = '')
                
            print("")
            for i in range(len(labels)):
                print("%5d" % labels[i], end = '\t')
                
                for val in cm[i]:                    
                    print("%5d \t " % val, end = '')
                print("")
            print("\n")

    return balanced_accuracy
        

# Evaluates the model and returns accuracy
def evaluate(model, X_test, Y_test, print_confusion_matrix = True, verbose = False):
    
    # TODO replace the accuracy with model.score of the sklearn
    
    predictions = model.predict(X_test)

    errors = np.count_nonzero(predictions != Y_test)
    accuracy = 1 - errors / X_test.shape[0]
        
    if verbose:    
        # Print some evaluation info
        print('Model Performance')
        print('errors=%d, accuracy=%.2f%%' % (errors, accuracy * 100))

        print("True Labels:")
        print(np.asarray(Y_test))
        print("Predictions:")
        print(predictions)

    if print_confusion_matrix:
        # If it is a binary classification
        if len(Counter(Y_test)) == 2:
    
            tn, fp, fn, tp = confusion_matrix(Y_test, predictions).ravel()        
            print("Confusion matrix")
            print("Correctly predicted 0: %d" % (tn))
            print("Correctly predicted 1: %d" % (tp))
            print("Incorrectly predicted 0: %d" % (fn))
            print("Incorrectly predicted 1: %d" % (fp))
            print("Accuracy for label 0: %d" % ((tn*100)/(tn+fn)))
            print("Accuracy for label 1: %d" % ((tp*100)/(tp+fp)))
        
        else:

            print("Confusion matrix")
            print("Horizontally - Predicted labels")
            print("Vertically - True labels")
            labels = list(range(len(Counter(Y_test))))
            cm = confusion_matrix(Y_test, predictions )

            print("      \t", end = '')
            for i in labels:
                print("%5d \t " % i, end = '')
                
            print("")
            for i in range(len(labels)):
                print("%5d" % labels[i], end = '\t')
                
                for val in cm[i]:                    
                    print("%5d \t " % val, end = '')
                print("")
            print("\n")

    return accuracy


# Prints out general statistics
def print_stats(total_runs, test_size):
    
    test_runs = round(total_runs * test_size)
    training_runs = total_runs - test_runs
    
    print("Test and training sets:")
    print("Trained on %d rows." % training_runs)
    print("Tested on %d rows." % test_runs)
    print("\n")


# Fit and test model n_runs times, find its mean. Uses Accuracy
def run_test_batch(model, data, test_size = 0.25, n_runs = 25, verbose=False):

    if verbose:
        print("Running a test batch of " , n_runs)
        print("Classifier: " + model.name)

    accuracy = []

    for i in range(n_runs):
        m = model(data, test_size)
        accuracy.append(m.accuracy*100)
        
    if verbose:
        print("Batch run check on the new dataset")
        print("Mean accuracy for",  n_runs ,"runs - ", np.mean(accuracy))
        print("Standart deviation for ",  n_runs ," runs -", np.std(accuracy), "\n")
        
    return np.mean(accuracy)


# Fit and test model n_runs times, find its mean. Uses balanced accuracy
def run_test_batch_ba(model, data, test_size = 0.25, n_runs = 25, verbose=False):

    if verbose:
        print("Running a test batch of " , n_runs)
        print("Classifier: " + model.name)

    balanced_accuracy = []

    for i in range(n_runs):
        m = model(data, test_size)
        
        predictions = m.model.predict(m.X_test)        
        ba = balanced_accuracy_score(m.Y_test, predictions)
        
        balanced_accuracy.append(ba*100)
        
    if verbose:
        print("Batch run check on the new dataset")
        print("Mean accuracy for",  n_runs ,"runs - ", np.mean(balanced_accuracy ))
        print("Standart deviation for ",  n_runs ," runs -", np.std(balanced_accuracy ), "\n")
        
    return np.mean(balanced_accuracy)


# Permutation importance - https://scikit-learn.org/stable/modules/permutation_importance.html
# Ranks and displays the importance of the features as per Classifier
def perm_importance(model, data):

    r = permutation_importance(model.model, model.X_test, model.Y_test)
    
    print("Feature \tImp mean \t Imp std")
    for i in r.importances_mean.argsort()[::-1]:
            print(f"{data.data.columns[i]:<8}\t"
                  f"{r.importances_mean[i]:.3f}\t\t"
                  f" +/- {r.importances_std[i]:.3f}")

