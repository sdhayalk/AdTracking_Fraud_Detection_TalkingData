import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, roc_auc_score
from feature_engineering import breakdown_datetime_into_columns

''' References:
	simple lightgbm example: https://github.com/Microsoft/LightGBM/blob/master/examples/python-guide/simple_example.py
'''

class Train:
	def __init__(self, X_train, y_train, X_validation, y_validation):
		pass

def basic_preprocessing(df):
	df = df.fillna(0)	# referred from: https://stackoverflow.com/questions/13295735/how-can-i-replace-all-the-nan-values-with-zeros-in-a-column-of-a-pandas-datafra?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
	
	for column_header in list(df.columns.values):	# referred from: https://stackoverflow.com/questions/19482970/get-list-from-pandas-dataframe-column-headers?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
		df = df[column_header].astype('category')			# referred from: https://pandas.pydata.org/pandas-docs/stable/categorical.html
	
	return df

def simple_demo():
	df = pd.read_csv('G:/DL/adtracking_fraud_detection/data/train.csv', nrows=10000000, parse_dates=['click_time'])
	df = breakdown_datetime_into_columns(df)
	df = basic_preprocessing(df)
	del df['click_time']
	del df['attributed_time']
	df_train, df_test = train_test_split(df, test_size=0.2)		# referred from: https://stackoverflow.com/questions/24147278/how-do-i-create-test-and-train-samples-from-one-dataframe-with-pandas?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa

	y_train = df_train['is_attributed']
	y_test = df_test['is_attributed']

	del df_train['is_attributed']
	del df_test['is_attributed']
	X_train = df_train
	X_test = df_test

	print(X_train.head(), "\n")
	print(y_train.head(), "\n")
	print(X_test.head(), "\n")
	print(y_test.head(), "\n")

	# create dataset for lightgbm
	lgb_train = lgb.Dataset(X_train, y_train)
	lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

	# specify your configurations as a dict
	params = {
	    'task': 'train',
	    'boosting_type': 'gbdt',
	    'is_unbalance': 'true',
	    'objective': 'binary',
	    'metric': {'auc'},
	    'num_leaves': 31,
	    'learning_rate': 0.05,
	    'feature_fraction': 0.9,
	    'bagging_fraction': 0.8,
	    'bagging_freq': 20,
	    'verbose': 0
	}

	print('Start training...')
	# train
	gbm = lgb.train(params,
	                lgb_train,
	                num_boost_round=20,
	                valid_sets=lgb_eval,
	                early_stopping_rounds=5)

	print('Save model...')
	# save model to file
	gbm.save_model('model.txt')

	print('Start predicting...')
	# predict
	y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
	# eval
	print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)
	print('auc', roc_auc_score(y_test, y_pred))

simple_demo()
