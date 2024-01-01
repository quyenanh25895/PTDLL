import joblib
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import auc, average_precision_score, confusion_matrix, f1_score, accuracy_score, precision_recall_curve, recall_score, precision_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

model = joblib.load('Models/svm_model_4.joblib')

data = pd.read_csv('Data/creditcard_2023.csv')

cols = data.columns
att_cols = cols[1:-1]
lab_cols = cols[-1]
print(att_cols)
data[att_cols] = preprocessing.MinMaxScaler((0, 1)).fit_transform(data[att_cols])
X = data[att_cols].values
y = data[lab_cols].values

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, shuffle=True)

y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# Hiển thị kết quả và đánh giá mô hình
print("Độ chính xác: {:.2f}%".format(accuracy * 100))
print("Nhớ lại: {:.2f}%".format(recall * 100))
print("Chính xác dự đoán: {:.2f}%".format(precision_score(y_test, y_pred) * 100))

# # Tính và hiển thị đồ thị Precision-Recall
average_precision = average_precision_score(y_test, y_pred)
precision, recall, curve = precision_recall_curve(y_test, y_pred)
plt.step(recall, precision, color='b', alpha=0.2, where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
plt.xlabel('Nhớ lại (Recall)')
plt.ylabel('Chính xác dự đoán (Precision)')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Đồ thị Precision-Recall: AP={:.2f}'.format(average_precision))
plt.show()

auc_pr = auc(recall, precision)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', label=f'Precision-Recall Curve (AUC = {auc_pr:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='best')
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure()
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xticks([0, 1], ['Predicted 0', 'Predicted 1'])
plt.yticks([0, 1], ['Actual 0', 'Actual 1'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


