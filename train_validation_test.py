import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from feature_engineering import breakdown_datetime_into_columns, basic_preprocessing

''' References:
	simple lightgbm example: https://github.com/Microsoft/LightGBM/blob/master/examples/python-guide/simple_example.py
'''

class TrainValidation:
	def __init__(self, df):
		self.df_train, self.df_test = train_test_split(df, test_size=0.2)		# referred from: https://stackoverflow.com/questions/24147278/how-do-i-create-test-and-train-samples-from-one-dataframe-with-pandas?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa

		self.y_train = self.df_train['is_attributed']
		self.y_test = self.df_test['is_attributed']

		del self.df_train['is_attributed']
		del self.df_test['is_attributed']
		self.X_train = self.df_train
		self.X_test = self.df_test

		print(self.X_train.head(), "\n")
		print(self.y_train.head(), "\n")
		print(self.X_test.head(), "\n")
		print(self.y_test.head(), "\n")


	def train_validation(self):
		self.lgb_train = lgb.Dataset(self.X_train, self.y_train)
		self.lgb_eval = lgb.Dataset(self.X_test, self.y_test, reference=self.lgb_train)

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
		self.gbm = lgb.train(params,
		                self.lgb_train,
		                num_boost_round=20,
		                valid_sets=self.lgb_eval,
		                early_stopping_rounds=5)

		print('Save model...')
		self.gbm.save_model('model.txt')		# save model to file

		print('Start predicting...')
		y_pred = self.gbm.predict(self.X_test, num_iteration=self.gbm.best_iteration)
		
		# evaluation
		print('The rmse of prediction is:', precision_recall_fscore_support(self.y_test, y_pred) ** 0.5)
		print('auc', roc_auc_score(self.y_test, y_pred))



df = pd.read_csv('G:/DL/adtracking_fraud_detection/data/train.csv', nrows=1000000, parse_dates=['click_time'], random_state=42)
df = breakdown_datetime_into_columns(df)
df = basic_preprocessing(df)
del df['click_time']
del df['attributed_time']

train_validation_instance = TrainValidation(df)
train_validation_instance.train_validation()
