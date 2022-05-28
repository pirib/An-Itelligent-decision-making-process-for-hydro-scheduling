from tpot import TPOTClassifier

import Data



# Get the data and split it
data = Data.Data(data_num = 7, command_column = 1, use_pre_process = True, include_std_dev = False, average_interval = 24)

X_train, X_test, y_train, y_test = data.Split()

# Get the TPOT for pipeline optimization
pipeline_optimizer = TPOTClassifier(generations=100, population_size=100, cv=5, verbosity=2)

pipeline_optimizer.fit(X_train, y_train)

# Export the findings
pipeline_optimizer.export('tpot_exported_pipeline.py')