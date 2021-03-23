import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from datetime import datetime

# Classifiers and Hyperparameters
first_clfs = {
    # 'LDA': (LinearDiscriminantAnalysis(), {'solver':['svd', 'lsqr'], 'tol':[0, 1e-3, 1e-4]}),
    # 'KNN': (KNeighborsClassifier(), {'n_neighbors': range(1, 21, 2), 'metric': ['euclidean', 'manhattan', 'minkowski'], 'weights': ['uniform', 'distance']}),
    # 'CART': (DecisionTreeClassifier(), {'criterion': ['gini', 'entropy'], 'ccp_alpha': [0, 1e-2, 12-4, 1e-6, 1e-8], 'class_weight': [None, 'balanced']}),
    'RF': (RandomForestClassifier(), {'n_estimators': [10, 100, 500, 1000], 'max_features': ['sqrt', 'log2'], 'bootstrap': [True, False],'max_depth': [3, 5, 10, 20, 50, 100, None]}),
    'SVM': (SVC(), {'kernel': ['poly', 'rbf', 'sigmoid', 'lienar'], 'C': [50, 10, 1.0, 0.1, 0.01], 'gamma':['scale', 'auto'], 'class_weight': ['balanced', None]}),
    'LR': (LogisticRegression(), {'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], 'C':[100, 10, 1.0, 0.1, 0.01], 'penalty': ['none', 'l1', 'l2', 'elasticnet']}),
}
# Add the two voting classifiers
clfs = first_clfs.copy()
# clfs['MVH'] = VotingClassifier(estimators=list(first_clfs.items()), voting='hard')
# # Remove SVM for MVS
# del first_clfs['SVM']
# clfs['MVS'] = VotingClassifier(estimators=list(first_clfs.items()), voting='soft')

# Get binary label
df = pd.read_pickle('full_colors.pkl')
df['Y'] = df.biota_percentage != 0

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(df[['B','V','R','I']], df['Y'], test_size=0.2, random_state=42)

# Scale data
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)

# Grid search
with open('hyperparameters.txt', 'a') as f:
    f.write(f'======================={datetime.now()}============================\n')
for name, (clf, grid) in clfs.items():
    try:
        gs = GridSearchCV(estimator=clf, param_grid=grid, cv=10, scoring='balanced_accuracy', n_jobs=-1, verbose=1)
        res = gs.fit(X_train, y_train)
        with open('hyperparameters.txt', 'a') as f:
            f.write(name + '\n')
            f.write("Best score: " + str(res.best_score_) + '\n')
            f.write(str(res.best_params_)+ '\n')
    except Exception as e:
        with open('hyperparameters.txt', 'a') as f:
            f.write(name + '\n')
            f.write(str(e) + '\n')