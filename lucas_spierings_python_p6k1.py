#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 12:48:09 2023

@author: lucas
"""
#%% Importing libraries for later use
import pandas as pd 
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier 
import matplotlib.pyplot as plt 
from sklearn import metrics, tree
from sklearn.preprocessing import binarize 
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import GridSearchCV 

#%% Assignment 1a
# Loading in the data
filepath = r"/Users/Filepath"
rawdata = pd.read_csv(filepath, sep=';') 

# Shows the amount of present values and datatype per column
print(rawdata.info())

# Prints the name of the column, the unique values in the column and the amount of times each value is present, nan is included
for col in rawdata:
    print('\n','Unique in column:', col)
    print(rawdata[col].value_counts(dropna=False).sort_index())

# Makes the plots containing information about the amount of nan values in the rawdata
nancount = pd.DataFrame(rawdata.isna().sum(), columns=['amount'])  
nancount['percentage'] = rawdata.isna().sum() / len(rawdata) *100 
nancount.sort_values(by='amount',ascending=False,inplace=True)

plot = nancount.plot(kind='bar',y='percentage',title='amount of missing data per column',ylabel='percentage') 
plot.axhline(y=40, color='black', linestyle=':')

#%% Assignment 1b
# Copies the rawdata and drops unneeded columns, also drops all rows containing at least 1 nan value
num_df = rawdata.copy()
num_df.drop(columns=['medicatie','zelfhulp','persoon_nr','gesl'],inplace=True)
num_df.dropna(inplace=True)

# Checks the amount of nan values per column
for col in num_df:
    getal = num_df[col].isna().sum()
    print("In kolom",col,"zitten",getal,"missende waarden.")

#%% Assignment 1c
# Encoder for the column 'diagnose'
diagnose_categories = ['gezond','ziek'] 
diagnose_encoder = OrdinalEncoder(categories=[diagnose_categories],handle_unknown='use_encoded_value', unknown_value=-1) 
num_df[['diagnose']] = diagnose_encoder.fit_transform(num_df[['diagnose']]) 

# Encoder for the column 'pijn_schaal'
pijn_schaal_categories = ['extreemlaag','laag','neutraal','hoog','extreemhoog'] 
pijn_schaal_encoder = OrdinalEncoder(categories=[pijn_schaal_categories],handle_unknown='use_encoded_value', unknown_value=-1) 
num_df[['pijn_schaal']] = pijn_schaal_encoder.fit_transform(num_df[['pijn_schaal']]) 

#%% Assignment 1d

# Makes function onehot, this function allows the user to take columns without a inherent sorting system and makes these into different columns
def onehot(df,feat): 
    encoder = OneHotEncoder(handle_unknown='ignore',sparse=False)     
    temp_df = pd.DataFrame(encoder.fit_transform(df[[feat]]),                             
    columns=encoder.get_feature_names_out())     
    df = pd.concat([df.reset_index(drop=True), temp_df], axis=1)     
    df.drop(columns=[feat], axis=1, inplace=True)     
    return df 

# Uses function onehot on the column pijn_type
num_df = onehot(num_df,'pijn_type') 

# Prints the information about the datatype per column, similair to assignment 1a
print(num_df.info())
# All colomns now only consist of the datatypes integer or float.

#%% Assignment 2a
# Splits num_df in X and y
X = num_df.copy().drop(columns=['diagnose'])
y = num_df[['diagnose']].copy()

# Splitting the dataset into a training set, validation set and testing set. 
X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.2, stratify=y, random_state=33)
X_train, X_val, y_train, y_val =  train_test_split(X_train, y_train, test_size=0.25, stratify = y_train, random_state=33) 

#%% Assignment 2b
# Defines and trains the model 'decisiontreeclassifier'
tree_model = DecisionTreeClassifier(criterion='entropy',max_depth=6,random_state=33) 
tree_model.fit(X_train, y_train) 
forest_model = RandomForestClassifier(criterion='entropy',max_depth=6,random_state=33) 
forest_model.fit(X_train, y_train) 

#%% Assignment 2c
# Plots the decision tree, in 'value' on the tree you see the balance between sick and healthy diagnoses 
# This has the smount of sick diagnoses being the value on the left and the amount of healthy diagnoses being the value on the left. 
# Under that there is the variable 'class', this one shows wether the sick diagnoses or the healthy diagnoses are more prevalent
plt.figure(figsize=(20,20)) 
tree.plot_tree(tree_model, feature_names=num_df['diagnose'],class_names=['gezond','ziek'], fontsize=10) 
plt.show()

#%% Assignment 2d
# Plots the first three random forests, in 'value' on the tree you see the balance between sick and healthy diagnoses 
# This has amount of sick diagnoses being the value on the left and the amount of healthy diagnoses being the value on the left. 
# Under that there is the variable 'class', this one shows wether the sick diagnoses or the healthy diagnoses are more prevalent
for i in range(3):
    plt.figure(figsize=(20,10))    
    tree.plot_tree(forest_model.estimators_[i],feature_names=num_df['diagnose'],class_names=['gezond','ziek'],fontsize = 5,filled=True)
    plt.show()

#%% Assignment 3a
y_pred_class_forest = forest_model.predict(X_val)
y_pred_prob_forest = forest_model.predict_proba(X_val)

print('De Confusion Matrix van het Random Forest model is:\n',metrics.confusion_matrix(y_val, y_pred_class_forest))

#%% Assignment 3b
print('De Accuracy van het Random Forest model is:',metrics.accuracy_score(y_val, y_pred_class_forest))  	
print('De Recall van het Random Forest model is:',metrics.recall_score(y_val, y_pred_class_forest))  	
print('De Precision van het Random Forest model is:',metrics.precision_score(y_val, y_pred_class_forest))
print('De F1-score van het Random Forest model is:',metrics.f1_score(y_val, y_pred_class_forest))

#%% Assignment 3c
# Defines fpr, tpr and thresholds
fpr, tpr, thresholds = metrics.roc_curve(y_val, y_pred_prob_forest[:,1]) 

# Makes ROC-curve plot with a random classification model
plt.plot(fpr, tpr) 
plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='k',    label='Random guess') 
plt.xlim([0.0, 1.0]) 
plt.ylim([0.0, 1.0]) 
plt.title('ROC curve for Randomforest classifier') 
plt.xlabel('False Positive Rate (1 - Specificity)') 
plt.ylabel('True Positive Rate (Recall, a.k.a. Sensitivity)') 
plt.grid(True) 

# Prints AUC of the current Random Forest model
print('De AUC van het Random Forest model is:', metrics.roc_auc_score(y_val, y_pred_prob_forest[:,1]) )  

#%% Assignment 3d

# Function to find the best Threshold manually
def evaluate_threshold(threshold): 
    print('TPR/Recall/Sensitivity:', tpr[thresholds > threshold][-1])      
    print ('FPR:', fpr[thresholds > threshold][-1])      
    print('Specificity:', 1 - fpr[thresholds > threshold][-1]) 

# Calls on the function evaluate_threshold
evaluate_threshold(0.05)
print("The best threshold would be 0.05, this is because your TPR is at 1 and your FPR is almost at 0 (0.03)")

#%% Assignment 3e

y_pred_class_forest = binarize([y_pred_prob_forest[:,1]], threshold=0.05)[0]   

print('De Confusion Matrix van het verbeterde Random Forest model is:\n',metrics.confusion_matrix(y_val, y_pred_class_forest))
print('De Accuracy van het verbeterde Random Forest model is:',metrics.accuracy_score(y_val, y_pred_class_forest))  	
print('De Recall van het verbeterde Random Forest model is:',metrics.recall_score(y_val, y_pred_class_forest))  	
print('De Precision van het verbeterde Random Forest model is:',metrics.precision_score(y_val, y_pred_class_forest))
print('De F1-score van het verbeterde Random Forest model is:',metrics.f1_score(y_val, y_pred_class_forest))

#%% Assignment 4a 

# Redefines the train and testsets
X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.2, stratify=y, random_state=33)

# Makes a new Decision tree model and trains it
model_tree2 = DecisionTreeClassifier(criterion='entropy',max_depth=6,) 
model_tree2.fit(X_train, y_train) 

# Cross validates 10 times
scores = cross_val_score(model_tree2, X_train, y_train, cv=10, scoring='roc_auc') 

# Prints the scores per cross-validation
for i in range(len(scores)):
    print("De AUC score van validatie",i+1,"is",scores[i])

# Prints the mean AUC-score
print("Hiervan is het gemiddelde van de AUC-scores:",np.mean(scores))

#%% Assignment 4b

# Makes k_range and uses Gridsearch to find the best max_depth for the decision tree model
k_range = list(range(1, 21)) 
param_grid = dict(max_depth=k_range) 
grid = GridSearchCV(model_tree2, param_grid, cv=10, scoring='roc_auc',n_jobs=-1) 
grid.fit(X_train, y_train) 
grid_df=pd.DataFrame(grid.cv_results_) 
grid_mean_scores = grid_df['mean_test_score']

# Prints the mean AUC-score per max depth
for i in range(len(grid_mean_scores)):
    print("De gemiddelde AUC-score van een 10 voudige crossvalidatie met een max_depth van ",i+1,"is",grid_mean_scores[i])

#%% Assignment 4c
# Plots the max_depth compared to the mean AUC-scores of a ten fold cross validation
plt.plot(k_range,grid_mean_scores,'b')
plt.ylabel('Cross validated AUC')
plt.xlabel('Value of k for max_depth for Decision Tree')
plt.show()

# Prints the best score, the best parameter and the best way to define the model.
print("Het best behaalde resultaat van max_depth is:",grid.best_score_)
print("Deze score is behaald met parameter:",grid.best_params_)
print("Dat maakt de beste manier om de decision tree te definieren het volgende:",grid.best_estimator_)










