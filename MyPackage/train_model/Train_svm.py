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

data = pd.read_csv('../../Data/creditcard_2023.csv')


def svm_joblib():
    cols = data.columns
    att_cols = cols[1:-1]
    lab_cols = cols[-1]
    print(att_cols)
    data[att_cols] = preprocessing.MinMaxScaler((0, 1)).fit_transform(data[att_cols])
    X = data[att_cols].values
    y = data[lab_cols].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, shuffle=True)

    model = SVC(kernel='linear', C=100)

    model.fit(X_train, y_train)

    joblib.dump(model, "Models/svm_model.joblib")
    X = pd.DataFrame(X)
    X.to_csv('Data/X.csv', index=False, header=False)
    y = pd.DataFrame(y)
    y.to_csv('Data/y.csv', index=False, header=False)
    print('Da train xong mo hinh')
    return X_train, X_test, y_train, y_test, model
