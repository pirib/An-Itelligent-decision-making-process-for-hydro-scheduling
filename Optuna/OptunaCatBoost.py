import Data
import Evaluation

import optuna

from catboost import CatBoostClassifier

# Optuna catboost https://github.com/optuna/optuna-examples/blob/main/catboost/catboost_simple.py


# Data settings
data_num = 9
debug = False
command_column = 1
test_size = 0.25

# Get the data
data = Data.Data(data_num = data_num, command_column = command_column, use_pre_process = False, include_std_dev = False, average_interval = 24)
X_train, X_test, y_train, y_test = data.Split()

# This is function we will try to optimize
def objective(trial):
    
    # Suggest current params
    params = {
        'iterations' : trial.suggest_int('iterations', 50, 300),    
        "objective": trial.suggest_categorical("objective", ["Logloss", "CrossEntropy"]),                     
        'depth' : trial.suggest_int('depth', 4, 10),                                           
        'random_strength' :trial.suggest_int('random_strength', 0, 100),                       
        'bagging_temperature' :trial.suggest_loguniform('bagging_temperature', 0.01, 100.00),
        "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
        'learning_rate' :trial.suggest_loguniform('learning_rate', 1e-3, 1e-1),
        "used_ram_limit": "4gb"
    }

    # Instantiate the model with given params
    model = CatBoostClassifier(
        eval_metric="AUC",
        l2_leaf_reg=50,
        border_count=64,
        verbose= False,
        **params
        )

    # Fit the model
    model.fit(X_train, y_train)

    # Get the accuracy
    accuracy = Evaluation.evaluate(model, X_test, y_test, False , False)

    return accuracy



# Create a study object and optimize the objective function.
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=500, n_jobs=8)
print(study.best_trial)


"""


"""