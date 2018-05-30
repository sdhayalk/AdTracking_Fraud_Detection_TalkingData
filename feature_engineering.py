import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

''' references:
	https://www.kaggle.com/graf10a/lightgbm-lb-0-9675
'''

def breakdown_datetime_into_columns(X):
	''' this function extracts the day, hour, minute, second from 'click_time' column and appending them as separate columns
	
	Arguments:
		X {pandas.core.frame.DataFrame} -- training features dataset
	
	Returns:
		X {pandas.core.frame.DataFrame} -- training features with the day, hour, minute, second in separate columns
	'''

	X['day'] = X['click_time'].dt.day.astype('uint8')
	X['hour'] = X['click_time'].dt.hour.astype('uint8')
	X['minute'] = X['click_time'].dt.minute.astype('uint8')
	X['second'] = X['click_time'].dt.second.astype('uint8')
	
	return X

def basic_preprocessing(df):
	df = df.fillna(0)	# referred from: https://stackoverflow.com/questions/13295735/how-can-i-replace-all-the-nan-values-with-zeros-in-a-column-of-a-pandas-datafra?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
	list_column_headers = list(df.columns.values)	# # referred from: https://stackoverflow.com/questions/19482970/get-list-from-pandas-dataframe-column-headers?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
	
	for column_header in list_column_headers:	
		df[column_header] = df[column_header].astype('category')	# referred from: https://pandas.pydata.org/pandas-docs/stable/categorical.html
	
	return df
	
	

def engineer_clicks_per_hour_by_same_ip(X):
	pass


def engineer_clicks_per_hour_by_same_channel(X):
	pass


def engineer_ip_cross_channel(X):
	pass


def engineer_num_of_channels_per_ip_per_day_per_hour(X):
	''' This function computes the number of channels associated with a given IP address and app
	
	Arguments:
		X {pandas.core.frame.DataFrame} -- features dataset
	
	Returns:
		X {pandas.core.frame.DataFrame}
	'''
	num_channels = X[['ip','day','hour','channel']]\
						.groupby(by=['ip', 'day', 'hour'])[['channel']]\
						.count()\
						.reset_index()\
						.rename(columns={'channel': 'n_channels'})
          
	# Merging the channels data with the main data set
	X = X.merge(num_channels, on=['ip','day','hour'], how='left')
	del num_channels
	return X


def engineer_num_of_channels_per_ip_per_app_per_os(X):
	''' This function computes the number of channels associated with a given ip, os, app
	
	Arguments:
		X {pandas.core.frame.DataFrame} -- features dataset
	
	Returns:
		X {pandas.core.frame.DataFrame}
	'''
	num_channels = X[['ip','app', 'os', 'channel']]\
						.groupby(by=['ip', 'app', 'os'])[['channel']]\
						.count()\
						.reset_index()\
						.rename(columns={'channel': 'ip_app_os_count'})

	# Merging the channels data with the main data set
	X = X.merge(num_channels, on=['ip','app', 'os'], how='left')
	del num_channels

# print("Adjusting the data types of the new count features... ")
# train.info()
# train['n_channels'] = train['n_channels'].astype('uint16')
# train['ip_app_count'] = train['ip_app_count'].astype('uint16')
# train['ip_app_os_count'] = train['ip_app_os_count'].astype('uint16')
