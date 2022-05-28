import Data
import Evaluation

import optuna

from sklearn.neural_network import MLPClassifier


# Data settings
data_num = 9
debug = False
command_column = 1
test_size = 0.25

# Get the data
data = Data.Data(data_num = data_num, 
                 command_column = command_column, 
                 use_pre_process = True, 
                 include_std_dev = True, 
                 average_interval = 4)

X_train, X_test, y_train, y_test = data.Split()

# This is function we will try to optimize
def objective(trial):
    
    # Suggest params
    param = {
        'activation': trial.suggest_categorical('activation', ['identity', 'logistic', 'tanh', 'relu']),
        'solver' : trial.suggest_categorical('solver', ['lbfgs', 'sgd', 'adam']),
        'hidden_layer_sizes' : trial.suggest_categorical('hidden_layer_sizes', [ (32),(64),(128), (256), (512), (32,32), (64,64)]),
        'alpha' : trial.suggest_loguniform('alpha', 0.0001, 0.01),
    }

    # Suggest params depending on the params chosen above
    if param['solver'] == 'sgd':
        param['learning_rate'] = trial.suggest_categorical('learning_rate', ('constant', 'invscaling', 'adaptive' ))
    if param['solver'] == 'sgd' or param['solver'] == 'adam':
        param['learning_rate_init'] = trial.suggest_loguniform('learning_rate_init', 0.0001, 0.01)
    
    if param['solver'] != 'lbfgs':
        param['batch_size'] = trial.suggest_int('batch_size', 1, 50)
            

    # Instantiate the model with given params
    model = MLPClassifier(*param)

    # Fit the model
    model.fit(X_train, y_train)

    # Get the accuracy
    accuracy = Evaluation.evaluate(model, X_test, y_test, False , False)

    return accuracy



# Create a study object and optimize the objective function.
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=200, n_jobs=-1)
print(study.best_trial)


