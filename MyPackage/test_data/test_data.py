import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.calibration import calibration_curve
from sklearn.svm import SVC
from sklearn.metrics import auc, classification_report, confusion_matrix, f1_score, accuracy_score, precision_recall_fscore_support, recall_score, precision_score, precision_recall_curve, average_precision_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, learning_curve, train_test_split
import joblib

data = pd.read_csv('/BTL/Data/creditcard.csv')


def data_test():
    cols = data.columns
    att_cols = cols[1:-1]
    lab_cols = cols[-1]
    print(att_cols)
    data[att_cols] = preprocessing.MinMaxScaler((0, 1)).fit_transform(data[att_cols])
    X_test = data[att_cols].values
    y_test = data[lab_cols].values
    print('Data_test')
    return X_test, y_test
