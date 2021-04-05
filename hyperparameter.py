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
clfs = {
    # 'LR': (LogisticRegression, {'C': np.logspace(-2, 0.5, 10), 'solver': ['liblinear'], 'penalty': ['l2']}),
    # 'LDA': (LinearDiscriminantAnalysis, {'solver':['svd'], 'tol': np.logspace(-6, -1, 10)),
    # 'KNN': (KNeighborsClassifier, {'n_neighbors': [31,41,51,61,71,81]}),
    # 'KNN': (KNeighborsClassifier, {'n_neighbors': np.logspace(1, 3, num=10, dtype=int), 'metric': ['euclidean']}),
    # 'CART': (DecisionTreeClassifier, {'criterion': ['gini'], 'ccp_alpha': [3e-5,5e-5,1e-5,5e-4]}),
    # 'SVM': (LinearSVC, {'C': [50, 100, 200]}),
    'RF': (RandomForestClassifier, {'n_estimators': [400, 600], 'min_samples_split': [50], 'class_weight': ['balanced'], 'ccp_alpha': [0.0003]}),
    }

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
SNs = np.random.uniform(low=1, high=100, size=X_val.shape[0])
# 2d array: SNs[realization][BVRI]
SNs = np.repeat(SNs, 4, axis=0).reshape(X_val.shape[0], 4)
X_noisy = X_val + np.random.normal(loc=0, scale=X_val/SNs)
while np.any(X_noisy < 0): # Avoid negative flux
    X_noisy[X_noisy < 0] = X_val[X_noisy < 0] + np.random.normal(loc=0, scale=X_val/SNs)
    
X_val = scaler.transform(X_noisy)

# Grid search
with open('hyperparameters.txt', 'a') as f:
    f.write(f'======================={datetime.now()}============================\n')
for name, (clf, grid) in clfs.items():
    with open('hyperparameters.txt', 'a') as f:
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
