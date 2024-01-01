# Import các thư viện
from MyPackage.train_model.Train_svm import *

X_train,X_test, y_train, y_test, model = svm_joblib()


y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# Hiển thị kết quả và đánh giá mô hình
print("Độ chính xác: {:.2f}%".format(accuracy * 100))
print("Nhớ lại: {:.2f}%".format(recall * 100))
print("Chính xác dự đoán: {:.2f}%".format(precision_score(y_test, y_pred) * 100))

# Hiển thị confusion matrix và báo cáo đánh giá mô hình
cm = np.array(confusion_matrix(y_test, y_pred, labels=[1, 0]))
confusion = pd.DataFrame(cm, index=['Gian lận', 'Bình thường'], columns=['Dự đoán là gian lận', 'Dự đoán là bình thường'])
print(confusion)
print(classification_report(y_test, y_pred))

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

