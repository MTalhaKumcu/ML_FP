# MEHMET_TALHA_KUMCU_38306
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


# Load the data from CSV file
data = pd.read_csv("final_test_classification_data.csv")  # PLS ENTER YOUR DATA PATH

# Split the data into train, test, and validation sets
# YOU CAN CHANGE YOUR test_size=, random_state and your dataset
train_data, test_data, train_labels, test_labels = train_test_split(
    data.iloc[:, :-1], data.iloc[:, -1], test_size=0.2, random_state=42
)  # Here
train_data, val_data, train_labels, val_labels = train_test_split(
    train_data, train_labels, test_size=0.25, random_state=42
)  # Here

scaler = StandardScaler()
train_data_scaled = scaler.fit_transform(train_data)
test_data_scaled = scaler.transform(test_data)
val_data_scaled = scaler.transform(val_data)

pca = PCA(n_components=2)
train_data_pca = pca.fit_transform(train_data_scaled)
test_data_pca = pca.transform(test_data_scaled)
val_data_pca = pca.transform(val_data_scaled)

knn_classifier = KNeighborsClassifier()
linear_svm_classifier = SVC(kernel="linear")
rbf_svm_classifier = SVC(kernel="rbf")
adaboost_classifier = AdaBoostClassifier()
naive_bayes_classifier = GaussianNB()
qda_classifier = QuadraticDiscriminantAnalysis()

knn_classifier.fit(train_data_pca, train_labels)
linear_svm_classifier.fit(train_data_pca, train_labels)
rbf_svm_classifier.fit(train_data_pca, train_labels)
adaboost_classifier.fit(train_data_pca, train_labels)
naive_bayes_classifier.fit(train_data_pca, train_labels)
qda_classifier.fit(train_data_pca, train_labels)

knn_val_acc = knn_classifier.score(val_data_pca, val_labels)
linear_svm_val_acc = linear_svm_classifier.score(val_data_pca, val_labels)
rbf_svm_val_acc = rbf_svm_classifier.score(val_data_pca, val_labels)
adaboost_val_acc = adaboost_classifier.score(val_data_pca, val_labels)
naive_bayes_val_acc = naive_bayes_classifier.score(val_data_pca, val_labels)
qda_val_acc = qda_classifier.score(val_data_pca, val_labels)

classifiers = ["k-NN", "Linear SVM", "RBF SVM", "AdaBoost", "Naive Bayes", "QDA"]
val_acc_scores = [
    knn_val_acc,
    linear_svm_val_acc,
    rbf_svm_val_acc,
    adaboost_val_acc,
    naive_bayes_val_acc,
    qda_val_acc,
]

best_model_index = val_acc_scores.index(max(val_acc_scores))
best_model = classifiers[best_model_index]
print(f"The best model based on validation accuracy is {best_model}.")
best_model_acc_test = None
if best_model == "k-NN":
    best_model_acc_test = knn_classifier.score(test_data_pca, test_labels)
elif best_model == "Linear SVM":
    best_model_acc_test = linear_svm_classifier.score(test_data_pca, test_labels)
elif best_model == "RBF SVM":
    best_model_acc_test = rbf_svm_classifier.score(test_data_pca, test_labels)
elif best_model == "AdaBoost":
    best_model_acc_test = adaboost_classifier.score(test_data_pca, test_labels)
elif best_model == "Naive Bayes":
    best_model_acc_test = naive_bayes_classifier.score(test_data_pca, test_labels)
elif best_model == "QDA":
    best_model_acc_test = qda_classifier.score(test_data_pca, test_labels)
# YOU CAN CHANGE YOUR labels = ["Validation", "Test"] => i mean names

labels = ["Validation", "Test"]  # Here
accuracy_scores = [val_acc_scores[best_model_index], best_model_acc_test]
plt.bar(labels, accuracy_scores)
plt.xlabel("Dataset")  # Here
plt.ylabel("Accuracy")  # Here
plt.title(f"Accuracy Comparison - {best_model}")  # Here
plt.show()


# MEHMET_TALHA_KUMCU_38306
