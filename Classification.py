import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE 


import warnings
warnings.filterwarnings("ignore")


class Classification:
    def __init__(self, smote=False):
        self.smote = smote
        self.data = self.readData()
        self.attrition = None
        self.model = None
        self.dataCleaning()
        self.preProcessing()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data, self.attrition, 
                                                                                test_size = 0.2)
       
        if self.smote:
            sm = SMOTE(random_state = 2) 
            self.X_train, self.y_train = sm.fit_sample(self.X_train, self.y_train)
        
    def readData(self):
        data = pd.read_csv('data/WA_Fn-UseC_-HR-Employee-Attrition.csv', encoding='utf-8-sig')
        return data
        
    def dataCleaning(self):
        #Drop columns irrelevant to our analysis
        drop_columns = ['EmployeeCount','EmployeeNumber','Over18','StandardHours']
        self.data = self.data.drop(drop_columns, axis=1)
        
        #drop_corr_columns=['Age','TotalWorkingYears','YearsAtCompany','NumCompaniesWorked']
        #self.data = self.data.drop(drop_corr_columns, axis=1)

        #Normalize the large income data for HourlyRate, DailyRate, MonthlyRate and MonthlyIncome
        norm_col = ['HourlyRate','DailyRate','MonthlyRate','MonthlyIncome']
        for col in norm_col:
            self.data[col] = (self.data[col] - self.data[col].mean())/ self.data[col].std()
            
    def preProcessing(self):
        #Convert categorical features to numerical features
        le = LabelEncoder()
        self.attrition = self.data['Attrition']
        self.attrition = le.fit_transform(self.attrition)
        self.data.drop('Attrition', axis=1, inplace=True)

        cat_col = self.data.select_dtypes(include=['object']).columns.values
        data_col = self.data[cat_col]
        self.data.drop(cat_col, axis=1, inplace=True)

        #One hot encode categorical data and combine dataframes
        data_col = pd.get_dummies(data_col)
        self.data = pd.concat([self.data, data_col], axis=1).values
        
    def buildANNModel(self, data):
        cl = Sequential()
        cl.add(Dense(20, activation='relu', kernel_initializer='random_normal',  input_shape=(data.shape[1],)))
        cl.add(Dropout(0.1))
        cl.add(Dense(10, activation='relu', kernel_initializer='random_normal',  input_shape=(data.shape[1],)))
        cl.add(Dropout(0.1))
        cl.add(Dense(5, activation='relu', kernel_initializer='random_normal',  input_shape=(data.shape[1],)))
        cl.add(Dropout(0.1))
        cl.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))
        adm = Adam()
        cl.compile(optimizer=adm, loss='binary_crossentropy', metrics=['accuracy'])
        return cl

    def trainANNModel(self, X_train, y_train, epochs=10, batch_size=10, validation_split=0.33, verbose=0):
        self.model = self.buildANNModel(X_train)
        self.model.fit(X_train, y_train, validation_split=validation_split, epochs=epochs, batch_size=batch_size, verbose=verbose)
        
    def evaluateANNModel(self,testX,testy):
        yhat_probs = self.model.predict(testX, verbose=0)
        yhat_classes = self.model.predict_classes(testX, verbose=0)
        yhat_probs = yhat_probs[:, 0]
        yhat_classes = yhat_classes[:, 0]

        accuracy = accuracy_score(testy, yhat_classes)
        print('Accuracy: %f' % accuracy)
        # precision tp / (tp + fp)
        precision = precision_score(testy, yhat_classes)
        print('Precision: %f' % precision)
        # recall: tp / (tp + fn)
        recall = recall_score(testy, yhat_classes)
        print('Recall: %f' % recall)
        # f1: 2 tp / (2 tp + fp + fn)
        f1 = f1_score(testy, yhat_classes)
        print('F1 score: %f' % f1)



