#!/usr/bin/env python
# coding: utf-8

import os
os.chdir('/home/guu8/emotion-classification/src')
import pandas as pd
#import seaborn as sns
#import matplotlib.pyplot as plt
import preprocess
#%%
print('Loading Data')
data = pd.read_csv('../data/Tweets.csv',encoding='latin-1')
print(data.shape)
#data = data.head(100)
#print('Visualize the Class Distribution in the Data')
#fig = plt.figure(figsize=(8,4))
#sns.barplot(x = data['Class'].unique(), y=data['Class'].value_counts())
#plt.show()
#print(data.shape)
#%%
data_matrix = preprocess.generate_data_matrix(data)
print('Writing the data matrix to CSV')
data_matrix.to_csv(r'../data/data_matrix.csv')

