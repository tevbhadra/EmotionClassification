#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 09:43:30 2019

@author: guu8
"""
import os
os.chdir('/home/guu8/emotion-classification/src')

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import preprocess

from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import hamming_loss
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support
from skmultilearn.model_selection import IterativeStratification
from sklearn.metrics import multilabel_confusion_matrix
from statistics import mean 
from xgboost import XGBClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
#from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
#%%
def transform_nrc_data(nrc_data):
    emotions_df = nrc_data.drop(['Sentence', 'negative','positive'], axis=1)
    emotions_df_no_label = emotions_df.loc[(emotions_df==0).all(axis=1)]
    emotions_df_with_label = emotions_df.loc[(emotions_df>0).any(axis=1)]
    #retain only dominant classes
    emotions_df_dominate_label = pd.DataFrame((emotions_df_with_label.T.values == np.amax(emotions_df_with_label.values, 1)).T*1, 
                                              columns = emotions_df_with_label.columns)
    emotions_df_dominate_label = emotions_df_dominate_label.set_index(emotions_df_with_label.index)
    frame = [emotions_df_dominate_label,emotions_df_no_label]
    emotions_new = pd.concat(frame).sort_index()
    return emotions_new
#%%
def visualize_label_distribution(emotions_new,categories):
    counts = []
    for i in categories:
        counts.append((i, emotions_new[i].sum()))
        df_stats = pd.DataFrame(counts, columns=['category', 'number_of_tweets'])
    print(df_stats)
    df_stats.plot(x='category', y='number_of_tweets', kind='bar', legend=False, grid=True, figsize=(8, 5))
    plt.title("Number of tweets per category")
    plt.ylabel('# of Occurrences', fontsize=12)
    plt.xlabel('Emotion Classes', fontsize=12)

    rowsums = emotions_new.sum(axis=1)
    x=rowsums.value_counts()#plot
    plt.figure(figsize=(8,5))
    ax = sns.barplot(x.index, x.values)
    plt.title("Multiple Labels")
    plt.ylabel('# of Occurrences', fontsize=12)
    plt.xlabel('# of Labels (emotion classes)', fontsize=12)
#%%
#Creating the Models
def get_model(classifier,x,y):
    if classifier=="SVM":
        model = OneVsRestClassifier(svm.SVC(gamma='auto'))
    elif classifier=="NB":
        model = OneVsRestClassifier(MultinomialNB(fit_prior=True, class_prior=None))
    elif classifier=="XGB":
        model = OneVsRestClassifier(XGBClassifier())
    elif classifier=="RF":
        model = OneVsRestClassifier(RandomForestClassifier())
    model.fit(x,y)
    return(model)
#%%
#Performing Predictions
def get_predictions(model,test):
    predictions = pd.DataFrame(model.predict(test))
    predictions.columns = categories
    return(predictions)
#%%
#Performance Evaluation
def performance_evaluation(y_test,predictions):
    perf = []
    
    subset_accuracy = accuracy_score(y_test,predictions)
    
#    for category in categories:
#        print('Test accuracy for {} is {}'.format(category,
#              accuracy_score(y_test,predictions)))
    
    hammingloss = hamming_loss(y_test,predictions)
    
    #confusion_matrix = multilabel_confusion_matrix(y_test,predictions)
#    for i in range(len(categories)):
#        print("Confusion Matrix for ",categories[i],"\n",confusion_matrix[i])
    #print(confusion_matrix)
    macro_res = precision_recall_fscore_support(y_test, predictions, average='macro')
    micro_res = precision_recall_fscore_support(y_test, predictions, average='micro')
    #weighted_res = precision_recall_fscore_support(y_test, predictions, average='weighted')
    #res = precision_recall_fscore_support(y_test, predictions)
 
    micro_avg_precision = micro_res[0]
    micro_avg_recall = micro_res[1]
    micro_avg_fscore = micro_res[2]
    
    macro_avg_precision = macro_res[0]
    macro_avg_recall = macro_res[1]
    macro_avg_fscore = macro_res[2]
    
    
    
    perf = [subset_accuracy,hammingloss,
            micro_avg_precision,micro_avg_recall,micro_avg_fscore,
            macro_avg_precision,macro_avg_recall,macro_avg_fscore]
    return(perf)
    
#%%
#cross-fold validation
def crossfold(n_rounds,n_splits,classifier,x,y):
    perf = []
    x_columns = x.columns
    y_columns = y.columns
    for i in range(n_rounds):
        print("Round: ",i)
        
        folds = IterativeStratification(n_splits=5, order=1)
        
        for train_index, test_index in folds.split(x, y):
            x = np.array(x)
            y = np.array(y)
            #print("TRAIN:", train_index, "TEST:", test_index)
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            x_train = pd.DataFrame(x_train)
            x_train.columns = x_columns
            x_test = pd.DataFrame(x_test)
            x_test.columns = x_columns
            
            y_train = pd.DataFrame(y_train)
            y_train.columns = y_columns
            y_test = pd.DataFrame(y_test)
            y_test.columns = y_columns
                
            if standardize == 1:
                scaler = StandardScaler()
                scaler.fit(x_train)
                x_train = scaler.transform(x_train)
                x_test = scaler.transform(x_test)
            print("Modelling")  
            model = get_model(classifier,x_train,y_train)
            print("Prediction")
            predictions = get_predictions(model,x_test)
            fold_perf = performance_evaluation(y_test,predictions)
            perf.append(fold_perf)
    return(perf)
#%%
classifier = "XGB"
standardize = 0
nrc_data = pd.read_csv('../data/NRC_output.csv',encoding='latin-1')
nrc_data.shape
emotions_new = transform_nrc_data(nrc_data)
categories = list(emotions_new.columns.values)
#visualize_label_distribution(emotions_new,categories)
Sentence = nrc_data['Sentence']
nrc_data_transformed = pd.concat([Sentence,emotions_new],axis=1)
#%%
nrc_data_transformed = nrc_data_transformed.sample(n=25000)
#%%
data_matrix = preprocess.generate_data_matrix(nrc_data_transformed)
#%%
nrc_data_transformed = nrc_data_transformed.reset_index(drop=True)
#%%
#data_matrix_with_labels = pd.concat([nrc_data_transformed,data_matrix],axis=1)
#data_matrix_with_labels = data_matrix_with_labels.drop(labels="Sentence",axis=1)
x = data_matrix
y = nrc_data_transformed.drop(labels="Sentence",axis=1)
#%%
nfolds = 5
nrounds=2
#%%
perf = crossfold(nrounds,nfolds,classifier,x,y)

mean_subset_accuracy = mean([ x[0] for x in perf])
mean_hamming_loss = mean([ x[1] for x in perf])
micro_avg_precision = mean([ x[2] for x in perf])
micro_avg_recall = mean([ x[3] for x in perf])
micro_avg_fscore = mean([ x[4] for x in perf])
macro_avg_precision = mean([ x[5] for x in perf])
macro_avg_recall = mean([ x[6] for x in perf])
macro_avg_fscore = mean([ x[7] for x in perf])

print("Mean Subset Accuracy = ",mean_subset_accuracy)
print("Mean Hamming Loss = ",mean_hamming_loss)
print("Micro Average Precision = ",micro_avg_precision)
print("Micro Average Recall = ",micro_avg_recall)
print("Micro Average F-Score = ",micro_avg_fscore)
print("Macro Average Precision = ",macro_avg_precision)
print("Macro Average Recall = ",macro_avg_recall)
print("Macro Average F-Score = ",macro_avg_fscore)

result = pd.DataFrame([mean_subset_accuracy,
                      mean_hamming_loss,
                      micro_avg_precision,
                      micro_avg_recall,
                      micro_avg_fscore,
                      macro_avg_precision,
                      macro_avg_recall,
                      macro_avg_fscore])
metrics = ["mean_subset_accuracy",
           "mean_hamming_loss",
           "micro_avg_precision",
           "micro_avg_recall",
           "micro_avg_fscore",
           "macro_avg_precision",
           "macro_avg_recall",
           "macro_avg_fscore"]
result.index = metrics
result.to_csv("../data/output/"+classifier+"_performance.csv")
#%%
x=x.drop(x.columns[0], axis=1)
y=y.drop(y.columns[0], axis=1)