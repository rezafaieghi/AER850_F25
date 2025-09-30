# Import sample data for binary classification
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X = data.data
y = data.target
print("Features shape:", X.shape)
print("Labels shape:", y.shape)


# Test-Train split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Build the first classifier using logistic regression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
clf1 = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, random_state=42))
])
clf1.fit(X_train, y_train)
print("Training accuracy:", clf1.score(X_train, y_train))
print("Test accuracy:", clf1.score(X_test, y_test))

# Evaluate the classifier using various metrics
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
y_pred_clf1 = clf1.predict(X_test)
cm_clf1 = confusion_matrix(y_test, y_pred_clf1)
print("Confusion Matrix:")
print(cm_clf1)
precision_clf1 = precision_score(y_test, y_pred_clf1)
recall_clf1 = recall_score(y_test, y_pred_clf1)
f1_clf1 = f1_score(y_test, y_pred_clf1)
print("Precision:", precision_clf1)
print("Recall:", recall_clf1)
print("F1 Score:", f1_clf1)


# Evaluating the effect of threshold values
import numpy as np
import matplotlib.pyplot as plt

y_scores = clf1.predict_proba(X_test)[:, 1]

thresholds = np.arange(0, 1.0, 0.1)
precisions_clf1 = []
recalls_clf1 = []
f1s_clf1 = []

for t in thresholds:
    y_pred_thr = (y_scores >= t).astype(int)
    precisions_clf1.append(precision_score(y_test, y_pred_thr))
    recalls_clf1.append(recall_score(y_test, y_pred_thr))
    f1s_clf1.append(f1_score(y_test, y_pred_thr))
    print(f"Threshold: {t:.1f}  Precision: {precisions_clf1[-1]:.3f}  Recall: {recalls_clf1[-1]:.3f}  F1: {f1s_clf1[-1]:.3f}")

plt.figure(figsize=(6,4))
plt.plot(thresholds, precisions_clf1, label="Precision")
plt.plot(thresholds, recalls_clf1, label="Recall")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Precision and Recall vs Threshold")
plt.legend()
plt.grid(True)
plt.show()



# Train and evaluate a classifier based on support vector machines
from sklearn.svm import SVC

clf2 = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", SVC(kernel="linear", probability=True, random_state=42))
])
clf2.fit(X_train, y_train)
print("SVM Training accuracy:", clf2.score(X_train, y_train))
print("SVM Test accuracy:", clf2.score(X_test, y_test))

y_pred_clf2 = clf2.predict(X_test)
cm_clf2 = confusion_matrix(y_test, y_pred_clf2)
print("SVM Confusion Matrix:")
print(cm_clf2)
precision_clf2 = precision_score(y_test, y_pred_clf2)
recall_clf2 = recall_score(y_test, y_pred_clf2)
f1_clf2 = f1_score(y_test, y_pred_clf2)
print("SVM Precision:", precision_clf2)
print("SVM Recall:", recall_clf2)
print("SVM F1 Score:", f1_clf2)

y_scores_clf2 = clf2.predict_proba(X_test)[:, 1]
precisions_clf2 = []
recalls_clf2 = []
f1s_clf2 = []
for t in thresholds:
    y_pred_thr = (y_scores_clf2 >= t).astype(int)
    p = precision_score(y_test, y_pred_thr)
    r = recall_score(y_test, y_pred_thr)
    f = f1_score(y_test, y_pred_thr)
    precisions_clf2.append(p); recalls_clf2.append(r); f1s_clf2.append(f)
    print(f"Threshold: {t:.1f}  Precision: {p:.3f}  Recall: {r:.3f}  F1: {f:.3f}")

plt.figure(figsize=(6,4))
plt.plot(thresholds, precisions_clf2, label="Precision (SVM)")
plt.plot(thresholds, recalls_clf2, label="Recall (SVM)")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("SVM: Precision and Recall vs Threshold")
plt.legend()
plt.grid(True)
plt.show()


# Train and evaluate a classifier based on decision trees
from sklearn.tree import DecisionTreeClassifier
clf3 = DecisionTreeClassifier(max_depth=4, random_state=42)
clf3.fit(X_train, y_train)
print("DT Training accuracy:", clf3.score(X_train, y_train))
print("DT Test accuracy:", clf3.score(X_test, y_test))

y_pred_clf3 = clf3.predict(X_test)
cm_clf3 = confusion_matrix(y_test, y_pred_clf3)
print("DT Confusion Matrix:")
print(cm_clf3)
precision_clf3 = precision_score(y_test, y_pred_clf3)
recall_clf3 = recall_score(y_test, y_pred_clf3)
f1_clf3 = f1_score(y_test, y_pred_clf3)
print("DT Precision:", precision_clf3)
print("DT Recall:", recall_clf3)
print("DT F1 Score:", f1_clf3)

y_scores_clf3 = clf3.predict_proba(X_test)[:, 1]
precisions_clf3, recalls_clf3, f1s_clf3 = [], [], []
for t in thresholds:
    y_pred_thr = (y_scores_clf3 >= t).astype(int)
    p = precision_score(y_test, y_pred_thr, zero_division=0)
    r = recall_score(y_test, y_pred_thr)
    f = f1_score(y_test, y_pred_thr)
    precisions_clf3.append(p); recalls_clf3.append(r); f1s_clf3.append(f)
    print(f"Threshold: {t:.1f}  Precision: {p:.3f}  Recall: {r:.3f}  F1: {f:.3f}")

plt.figure(figsize=(6,4))
plt.plot(thresholds, precisions_clf3, label="Precision (DT)")
plt.plot(thresholds, recalls_clf3, label="Recall (DT)")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Decision Tree: Precision and Recall vs Threshold")
plt.legend()
plt.grid(True)
plt.show()


# Train and evaluate a classifier based on random forests
from sklearn.ensemble import RandomForestClassifier
clf4 = RandomForestClassifier(n_estimators=200, random_state=42)
clf4.fit(X_train, y_train)
print("RF Training accuracy:", clf4.score(X_train, y_train))
print("RF Test accuracy:", clf4.score(X_test, y_test))

y_pred_clf4 = clf4.predict(X_test)
cm_clf4 = confusion_matrix(y_test, y_pred_clf4)
print("RF Confusion Matrix:")
print(cm_clf4)
precision_clf4 = precision_score(y_test, y_pred_clf4)
recall_clf4 = recall_score(y_test, y_pred_clf4)
f1_clf4 = f1_score(y_test, y_pred_clf4)
print("RF Precision:", precision_clf4)
print("RF Recall:", recall_clf4)
print("RF F1 Score:", f1_clf4)

y_scores_clf4 = clf4.predict_proba(X_test)[:, 1]
precisions_clf4, recalls_clf4, f1s_clf4 = [], [], []
for t in thresholds:
    y_pred_thr = (y_scores_clf4 >= t).astype(int)
    p = precision_score(y_test, y_pred_thr, zero_division=0)
    r = recall_score(y_test, y_pred_thr)
    f = f1_score(y_test, y_pred_thr)
    precisions_clf4.append(p); recalls_clf4.append(r); f1s_clf4.append(f)
    print(f"Threshold: {t:.1f}  Precision: {p:.3f}  Recall: {r:.3f}  F1: {f:.3f}")

plt.figure(figsize=(6,4))
plt.plot(thresholds, precisions_clf4, label="Precision (RF)")
plt.plot(thresholds, recalls_clf4, label="Recall (RF)")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Random Forest: Precision and Recall vs Threshold")
plt.legend()
plt.grid(True)
plt.show()




# Evalauting classifiers based on ROC Curves
from sklearn.metrics import roc_curve, roc_auc_score

models = [
    ("Logistic", clf1),
    ("SVM", clf2),
    ("DT", clf3),
    ("RF", clf4),
]

plt.figure(figsize=(6,4))
for name, model in models:
    y_scores = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    auc = roc_auc_score(y_test, y_scores)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.grid(True)
plt.show()




# Using grid search to find the classifer with the best AUC
# Followed by fine-tuning threshold values for the best models.
# First, a grid search for support vector machine
from sklearn.model_selection import GridSearchCV

# SVC grid search → clf5
pipe1 = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", SVC(probability=True, random_state=42))
])
param1 = {
    "clf__kernel": ["linear", "rbf"],
    "clf__C": [0.1, 1, 10],
    "clf__gamma": ["scale"]
}
gs1 = GridSearchCV(pipe1, param_grid=param1, scoring="roc_auc", cv=5, n_jobs=-1)
gs1.fit(X_train, y_train)
clf5 = gs1.best_estimator_
print("SVM Training accuracy:", clf5.score(X_train, y_train))
print("SVM Test accuracy:", clf5.score(X_test, y_test))

y_pred_clf5 = clf5.predict(X_test)
cm_clf5 = confusion_matrix(y_test, y_pred_clf5)
print("SVM Confusion Matrix:")
print(cm_clf5)
precision_clf5 = precision_score(y_test, y_pred_clf5)
recall_clf5 = recall_score(y_test, y_pred_clf5)
f1_clf5 = f1_score(y_test, y_pred_clf5)
print("SVM Precision:", precision_clf5)
print("SVM Recall:", recall_clf5)
print("SVM F1 Score:", f1_clf5)

y_scores_clf5 = clf5.predict_proba(X_test)[:, 1]
precisions_clf5, recalls_clf5, f1s_clf5 = [], [], []
for t in thresholds:
    y_pred_thr = (y_scores_clf5 >= t).astype(int)
    p = precision_score(y_test, y_pred_thr, zero_division=0)
    r = recall_score(y_test, y_pred_thr)
    f = f1_score(y_test, y_pred_thr)
    precisions_clf5.append(p); recalls_clf5.append(r); f1s_clf5.append(f)
    print(f"Threshold: {t:.1f}  Precision: {p:.3f}  Recall: {r:.3f}  F1: {f:.3f}")

plt.figure(figsize=(6,4))
plt.plot(thresholds, precisions_clf5, label="Precision (SVM)")
plt.plot(thresholds, recalls_clf5, label="Recall (SVM)")
plt.xlabel("Threshold"); plt.ylabel("Score"); plt.title("SVM: Precision and Recall vs Threshold")
plt.legend(); plt.grid(True); plt.show()


# Second, a grid search for random forest
from sklearn.ensemble import RandomForestClassifier
pipe2 = RandomForestClassifier(random_state=42)
param2 = {
    "n_estimators": [200, 400, 600],
    "max_depth": [None, 6, 10, 16],
    "min_samples_leaf": [1, 2, 4]
}
gs2 = GridSearchCV(pipe2, param_grid=param2, scoring="roc_auc", cv=5, n_jobs=-1)
gs2.fit(X_train, y_train)
clf6 = gs2.best_estimator_
print("RF Training accuracy:", clf6.score(X_train, y_train))
print("RF Test accuracy:", clf6.score(X_test, y_test))

y_pred_clf6 = clf6.predict(X_test)
cm_clf6 = confusion_matrix(y_test, y_pred_clf6)
print("RF Confusion Matrix:")
print(cm_clf6)
precision_clf6 = precision_score(y_test, y_pred_clf6)
recall_clf6 = recall_score(y_test, y_pred_clf6)
f1_clf6 = f1_score(y_test, y_pred_clf6)
print("RF Precision:", precision_clf6)
print("RF Recall:", recall_clf6)
print("RF F1 Score:", f1_clf6)

y_scores_clf6 = clf6.predict_proba(X_test)[:, 1]
precisions_clf6, recalls_clf6, f1s_clf6 = [], [], []
for t in thresholds:
    y_pred_thr = (y_scores_clf6 >= t).astype(int)
    p = precision_score(y_test, y_pred_thr, zero_division=0)
    r = recall_score(y_test, y_pred_thr)
    f = f1_score(y_test, y_pred_thr)
    precisions_clf6.append(p); recalls_clf6.append(r); f1s_clf6.append(f)
    print(f"Threshold: {t:.1f}  Precision: {p:.3f}  Recall: {r:.3f}  F1: {f:.3f}")

plt.figure(figsize=(6,4))
plt.plot(thresholds, precisions_clf6, label="Precision (RF)")
plt.plot(thresholds, recalls_clf6, label="Recall (RF)")
plt.xlabel("Threshold"); plt.ylabel("Score"); plt.title("RF: Precision and Recall vs Threshold")
plt.legend(); plt.grid(True); plt.show()


models = [
    ("Logistic", clf1),
    ("SVM", clf2),
    ("DT", clf3),
    ("RF", clf4),
    ("SVM - GridSearched", clf5),
    ("RF - GridSearched", clf6),
]

plt.figure(figsize=(6,4))
for name, model in models:
    y_scores = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    auc = roc_auc_score(y_test, y_scores)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.grid(True)
plt.show()


