import lightgbm as lgb

import Data


# Get the data
data = Data.Data(data_num = 9, command_column = 1, use_pre_process = True, include_std_dev = True, average_interval = 4, normalize = True, debug = False)

# Split
X_train, X_test, y_train, y_test = data.Split()

# Train/Test data
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test)


# Train model

parameters = {
    'objective': 'binary',
    'metric': 'auc',
    'is_unbalance': 'true',
    'boosting': 'gbdt',
    'num_leaves': 31,
    'feature_fraction': 0.5,
    'bagging_fraction': 0.5,
    'bagging_freq': 20,
    'learning_rate': 0.05,
    'verbose': 1
}

model = lgb.train(  parameters,
                    train_data,
                    valid_sets=test_data,
                    num_boost_round=5000,
                    early_stopping_rounds=200)

prediction = model.predict(X_test, num_iteration=model.best_iteration)