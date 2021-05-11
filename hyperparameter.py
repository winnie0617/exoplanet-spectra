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

from util import *

# Classifiers and Hyperparameters
clfs = {
    # 'LR': (LogisticRegression, {'C': np.logspace(-1, 1, 5), 'solver': ['liblinear'], 'penalty': ['l2', 'l1']}),
    # 'LDA': (LinearDiscriminantAnalysis, {'solver':['svd'], 'tol': np.logspace(-10, 2, 10)}),
    # 'KNN': (KNeighborsClassifier, {'n_neighbors': [201, 1501], 'weights':['distance', 'uniform']}),
    # 'CART': (DecisionTreeClassifier, {'criterion': ['gini'], 'ccp_alpha': [1e-5], 'min_samples_split': [2, 5, 20, 5, 100, 500, 1000]}),
    'SVM': (LinearSVC, {'C': [1e-6],'dual':[False]}),
    # 'RF': (RandomForestClassifier, {'n_estimators': [100], 'max_depth': [None], 'class_weight': [None], 'ccp_alpha': [0.0003]}),
    }

# Get binary label
df = pd.read_pickle('full_colors.pkl')
df['Y'] = df.biota_percentage != 0

# Get train-val set
X_train_val, X_test, y_train_val, y_test = train_test_split(df[['B','V','R','I']], df['Y'], test_size=0.2, random_state=42)
# Split into train and val
X_train, X_val, y_train, y_val = train_test_split(df[['B','V','R','I']], df['Y'], test_size=0.2, random_state=42)

# Add noise and aggregate negative samples
X_train, y_train = upsample_minority(X_train, y_train)
X_train = add_noise(X_train)

# Scale train
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)

X_val = add_noise(X_val.to_numpy())
X_val = scaler.transform(X_val)

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
