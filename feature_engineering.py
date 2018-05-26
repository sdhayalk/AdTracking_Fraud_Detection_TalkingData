import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def breakdown_datetime_into_columns(X_train):
	''' this function extracts the day, hour, minute, second from 'click_time' column and appending them as separate columns
	
	Arguments:
		X_train {pandas.core.frame.DataFrame} -- training features dataset
	
	Returns:
		X_train {pandas.core.frame.DataFrame} -- training features with the day, hour, minute, second in separate columns
	'''

	X_train['day'] = X_train['click_time'].dt.day.astype('uint8')
	X_train['hour'] = X_train['click_time'].dt.hour.astype('uint8')
	X_train['minute'] = X_train['click_time'].dt.minute.astype('uint8')
	X_train['second'] = X_train['click_time'].dt.second.astype('uint8')
	print(type(X_train))
	
	return X_train

# load subset of the training data
X_train = pd.read_csv('G:/DL/adtracking_fraud_detection/data/train.csv', nrows=1000000, parse_dates=['click_time'])
X_train = breakdown_datetime_into_columns(X_train)

# Show the head of the table
print(X_train.head())