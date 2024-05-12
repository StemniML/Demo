from flask import Flask, render_template, request
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Cairo')

app = Flask(__name__)

def get_user_input():
    num_features = 20
    user_data = []
    for i in range(num_features):
        user_input = request.form.get(f"feature_{i+1}")
        if user_input is None or user_input == "":
            user_data.append(None)
        else:
            user_data.append(float(user_input))

        # Get toggle value
        toggle_value = int(request.form.get(f"toggle_feature_{i+1}", 0))
        user_data.append(toggle_value)

    return user_data

def as_percent(value):
    return "{:.2%}".format(value)

def calculate_auc_with_plot(X_train, y_train, X_test, y_test, classifier):
    classifier.fit(X_train, y_train)

    mean_fpr = np.linspace(0, 1, 100)
    tprs_train = []
    aucs_train = []
    tprs_test = []
    aucs_test = []

    for train_idx, test_idx in StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X_train, y_train):
        X_train_fold, X_val_fold = X_train[train_idx], X_train[test_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[test_idx]

        classifier.fit(X_train_fold, y_train_fold)

        probas_train = classifier.predict_proba(X_train_fold)[:, 1]
        fpr_train, tpr_train, _ = roc_curve(y_train_fold, probas_train)
        tprs_train.append(np.interp(mean_fpr, fpr_train, tpr_train))
        roc_auc_train = roc_auc_score(y_train_fold, probas_train)
        aucs_train.append(roc_auc_train)

        probas_test = classifier.predict_proba(X_test)[:, 1]
        fpr_test, tpr_test, _ = roc_curve(y_test, probas_test)
        tprs_test.append(np.interp(mean_fpr, fpr_test, tpr_test))
        roc_auc_test = roc_auc_score(y_test, probas_test)
        aucs_test.append(roc_auc_test)

    mean_tpr_train = np.mean(tprs_train, axis=0)
    mean_tpr_train[0] = 0.0
    mean_tpr_train[-1] = 1.0
    mean_auc_train = np.mean(aucs_train)

    mean_tpr_test = np.mean(tprs_test, axis=0)
    mean_tpr_test[0] = 0.0
    mean_tpr_test[-1] = 1.0
    mean_auc_test = np.mean(aucs_test)

    return mean_fpr, mean_tpr_train, mean_auc_train, mean_tpr_test, mean_auc_test

@app.route('/', methods=['GET', 'POST'])
def index():
    missing_data = False
    user_inputs = [None] * 20
    toggle_values = [0] * 20
    user_proba_percent = None
    auc_score_train_rf = None
    auc_score_test_rf = None
    auc_score_train_svm = None
    auc_score_test_svm = None
    auc_score_train_lr = None
    auc_score_test_lr = None
    plt_path = None
    user_prediction = None

    if request.method == 'POST':
        user_input_data = get_user_input()

        if None in user_input_data[::2]: # Check only non-toggle values for missing data
            missing_data = True
        else:
            X, y = make_classification(
                n_samples=1000,
                n_features=20,
                n_classes=2,
                random_state=42
            )

            # Split the data into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            svc = SVC(kernel="linear")
            rfe = RFE(estimator=svc, n_features_to_select=10, step=1)
            rfe.fit(X_train, y_train)
            selected_features = np.where(rfe.support_)[0]
            X_train_selected = X_train[:, selected_features]
            X_test_selected = X_test[:, selected_features]
            user_inputs = np.array(user_input_data[::2])  # Only non-toggle values
            toggle_values = np.array(user_input_data[1::2])  # Toggle values

            # Random Forest
            rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
            mean_fpr_rf, mean_tpr_train_rf, auc_score_train_rf, mean_tpr_test_rf, auc_score_test_rf = calculate_auc_with_plot(
                X_train_selected, y_train, X_test_selected, y_test, rf_classifier)

            # SVM
            svm_classifier = SVC(kernel="linear", probability=True, random_state=42)
            mean_fpr_svm, mean_tpr_train_svm, auc_score_train_svm, mean_tpr_test_svm, auc_score_test_svm = calculate_auc_with_plot(
                X_train_selected, y_train, X_test_selected, y_test, svm_classifier)

            # Logistic Regression
            lr_classifier = LogisticRegression(solver='liblinear', random_state=42)
            mean_fpr_lr, mean_tpr_train_lr, auc_score_train_lr, mean_tpr_test_lr, auc_score_test_lr = calculate_auc_with_plot(
                X_train_selected, y_train, X_test_selected, y_test, lr_classifier)

            user_input_selected = user_inputs[selected_features].reshape(1, -1) # Only non-toggle values

            user_proba_rf = rf_classifier.predict_proba(user_input_selected)[:, 1]
            user_proba_percent = as_percent(user_proba_rf[0])

            # Determine user prediction for Random Forest
            threshold = 0.7  # You can adjust this threshold if needed
            user_prediction = "Positive" if user_proba_rf >= threshold else "Negative"

            # Plot ROC curve for all models
            plt.figure(figsize=(8, 6))

            # Random Forest
            plt.plot(mean_fpr_rf, mean_tpr_test_rf, color='darkorange', lw=2, label='RF Test AUC = %0.2f' % auc_score_test_rf)
            #plt.plot(mean_fpr_rf, mean_tpr_train_rf, color='green', lw=2, label='RF Train AUC = %0.2f' % auc_score_train_rf)

            # SVM
            plt.plot(mean_fpr_svm, mean_tpr_test_svm, color='blue', lw=2, label='SVM Test AUC = %0.2f' % auc_score_test_svm)
            #plt.plot(mean_fpr_svm, mean_tpr_train_svm, color='red', lw=2, label='SVM Train AUC = %0.2f' % auc_score_train_svm)

            # Logistic Regression
            plt.plot(mean_fpr_lr, mean_tpr_test_lr, color='purple', lw=2, label='LR Test AUC = %0.2f' % auc_score_test_lr)
            #plt.plot(mean_fpr_lr, mean_tpr_train_lr, color='yellow', lw=2, label='LR Train AUC = %0.2f' % auc_score_train_lr)

            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC)')
            plt.legend(loc="lower right")

            # Add user prediction to the plot
            user_prediction_label = 1 if user_prediction == "Positive" else 0
            plt.scatter(mean_tpr_test_rf[user_prediction_label], mean_tpr_test_rf[user_prediction_label], color='red', label='User Input')

            plt_path = "/static/roc_curve.png"
            plt.savefig('.' + plt_path)
            plt.close()

    return render_template('index.html', missing_data=missing_data, user_inputs=user_inputs, toggle_values=toggle_values, user_proba=user_proba_percent,
                           auc_score_train_rf=auc_score_train_rf, auc_score_test_rf=auc_score_test_rf,
                           auc_score_train_svm=auc_score_train_svm, auc_score_test_svm=auc_score_test_svm,
                           auc_score_train_lr=auc_score_train_lr, auc_score_test_lr=auc_score_test_lr,
                           plt_path=plt_path, user_prediction=user_prediction)

if __name__ == '__main__':
    app.run(debug=True)
