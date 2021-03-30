import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from datetime import datetime

# Classifiers and Hyperparameters
first_clfs = {
    'LDA': (LinearDiscriminantAnalysis, {'solver':['svd', 'lsqr'], 'tol':[1, 0, 0.5, 1e-1, 1e-2]}),
    'KNN': (KNeighborsClassifier, {'n_neighbors': range(1, 21, 2), 'metric': ['euclidean', 'manhattan', 'minkowski'], 'weights': ['uniform', 'distance']}),
    'CART': (DecisionTreeClassifier, {'criterion': ['gini', 'entropy'], 'ccp_alpha': [0, 1e-2, 12-4, 1e-6, 1e-8], 'class_weight': [None, 'balanced']}),
    # 'RF': (RandomForestClassifier, {'n_estimators': [10, 100, 500, 1000], 'max_features': ['sqrt', 'log2'], 'bootstrap': [True, False],'max_depth': [3, 5, 10, 20, 50, 100, None]}),
    # 'SVM': (SVC, {'kernel': ['poly', 'rbf', 'sigmoid', 'lienar']}), #, 'C': [50, 10, 1.0, 0.1, 0.01], 'gamma':['scale', 'auto'], 'class_weight': ['balanced', None]}),
    'LR': (LogisticRegression, {'C':[1.0, 0.1, 0.01]}),
    # 'LR': (LogisticRegression, {'C':[100, 10, 1.0, 0.1, 0.01], 'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], 'penalty': ['none', 'l1', 'l2', 'elasticnet']}),
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

# Get train-val set
X_train_val, X_test, y_train_val, y_test = train_test_split(df[['B','V','R','I']], df['Y'], test_size=0.2, random_state=42)
# Split into train and val
X_train, X_val, y_train, y_val = train_test_split(df[['B','V','R','I']], df['Y'], test_size=0.2, random_state=42)
# Scale train
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
# Add noise to val set
np.random.seed(42) # Reproducible result
SNs = np.random.uniform(low=1, high=1, size=X_val.shape[0])
# 2d array: SNs[realization][BVRI]
SNs = np.repeat(SNs, 4, axis=0).reshape(X_val.shape[0], 4)
X_noisy = X_val + np.random.normal(loc=0, scale=X_val/SNs)
while np.any(X_noisy < 0): # Avoid negative flux
    X_noisy[X_noisy < 0] = X_val[X_noisy < 0] + np.random.normal(loc=0, scale=X_val/SNs)
    
X_val = scaler.transform(X_noisy)

# # Train and val for different hyperparameters
# for name, clf in clfs.items():
#     print(name)
#     clf.fit(X_train, y_train)

#     # Save accuracy for each model
#     res = {'S/N': SN}
#     for name, clf in clfs.items():
#         y_pred = clf.predict(X_scaled)
#         res[name] = balanced_accuracy_score(y_test, y_pred)
#     print(res)
#     df = df.append(res, ignore_index=True)

# Grid search
with open('hyperparameters.txt', 'a') as f:
    f.write(f'======================={datetime.now()}============================\n')
    for name, (clf, grid) in clfs.items():
        print(name)
        f.write('\n'+ name + '\n')
        combos = list(ParameterGrid(grid))
        best_score = 0
        best_params = {}

        for combo in combos:
            curr = clf(**combo)
            curr.fit(X_train, y_train)
            y_pred = curr.predict(X_val)
            acc = round(balanced_accuracy_score(y_val, y_pred),4)
            f.write(f'{acc}: {combo}\n')
            if acc > best_score:
                best_score, best_params = acc, combo
    
        f.write("Best score: " + str(best_score) + '\n')
        f.write("Best params " + str(best_params) + '\n')
