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
	
	return X_train

def basic_preprocessing(df):
	df = df.fillna(0)	# referred from: https://stackoverflow.com/questions/13295735/how-can-i-replace-all-the-nan-values-with-zeros-in-a-column-of-a-pandas-datafra?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
	list_column_headers = list(df.columns.values)	# # referred from: https://stackoverflow.com/questions/19482970/get-list-from-pandas-dataframe-column-headers?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
	
	for column_header in list_column_headers:	
		df[column_header] = df[column_header].astype('category')	# referred from: https://pandas.pydata.org/pandas-docs/stable/categorical.html
	
	return df
	
	
def convert_features_to_categorical(X, features):
	pass


def engineer_clicks_per_hour_by_same_ip(X_train):
	pass


def engineer_clicks_per_hour_by_same_channel(X_train):
	pass


def engineer_ip_cross_channel(X_train):
	pass


# load subset of the training data
# X_train = pd.read_csv('G:/DL/adtracking_fraud_detection/data/train.csv', nrows=1000000, parse_dates=['click_time'])
# X_train = breakdown_datetime_into_columns(X_train)

# Show the head of the table
# print(X_train.head())
# print(X_train.describe())