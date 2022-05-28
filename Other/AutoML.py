"""
To run this script one needs Python 3.5 or higher and a Linux system (or using a WSL - check the second link)

Auto sklearn - https://automl.github.io/auto-sklearn/master/
To run this on windows - https://medium.com/@gracy.f/automl-for-python-on-windows-314ca8ea6955

"""

import autosklearn.classification

import sklearn.model_selection

import sklearn.metrics

import Data

import time



start = time.time()

print("Starting AutoML")

# Get the data
data = Data.Data(data_num = 7, command_column = 1, 
                      use_pre_process = True, include_std_dev = False)

# Split the data for training/testing
X_train, X_test, y_train, y_test = data.Split()

# Start AutoML
automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task = 3600 * 8 # run for 8 hours
    )

automl.fit(X_train, y_train)

try:
    automl.cv_results.to_csv()
    automl.performance_over_time.to_csv()
    print("created csv files")
except:
    print("cv results ")


try:
    automl.cv_results_.to_csv()
    automl.performance_over_time_.to_csv()
except:
    print("sv results _")

try:
    print(automl.leaderboard())
except:
    print("Leaderboard doesnt work")    

try:
    print(automl.show_models())
except:
    print("Show models doesnt work")

predictions = automl.predict(X_test)

print("Accuracy score:", sklearn.metrics.accuracy_score(y_test, predictions))


end = time.time()
print(end - start)

# started at 15:36