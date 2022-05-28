import optuna

import Data
import Evaluation
import RandomForest


def find_best_data_pre_process(data_num, command_column, horizon):
    
    def objective(trial):
    
        pre_process = trial.suggest_categorical("pre_process", [True, False]) 
        
        include_std_dev = False
        average_interval = 1
        # The pre process routine
        if pre_process:
            
            # Whether or not to include std deviation
            include_std_dev = trial.suggest_categorical("std", [True, False]) 

            # Average interval
            if horizon == 168:
                average_interval = trial.suggest_categorical("av int", [1,2, 4, 8, 12, 24,56, 84, 168])    # For 168 hour horizons
            elif horizon == 72:
                average_interval = trial.suggest_categorical("av int", [1, 2, 4, 8, 12, 24, 36, 72])         # For 72 hour
            elif horizon == 210:
                average_interval = trial.suggest_categorical("av int", [1, 3, 5, 10, 21, 30, 70, 210])    # For 210 hour
            elif horizon == 240:
                average_interval = trial.suggest_categorical("av int", [1, 2, 4, 8, 10, 24, 48, 120, 240])    # For 210 hour
            else:
                print("horizon got an unexpected value")
                raise()


        # Get the data
        data = Data.Data(data_num = data_num, command_column = command_column, 
                         
                         # Pre process
                         use_pre_process = True,            
                         include_std_dev = include_std_dev, 
                         average_interval = average_interval,
                         
                         )
                
        accuracy = Evaluation.run_test_batch(RandomForest.RF, data, 0.25, n_runs = 1)
        print(accuracy)
        return accuracy
    
    
    # Create a study object and optimize the objective function.
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)

    print("Dataset number: %1i" % data_num)    
    print("Command number: %1i" % command_column)    
    print(study.best_trial)


find_best_data_pre_process(11, 1, 240)

