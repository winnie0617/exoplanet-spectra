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
SN_ranges = [(0, 30), (30, 70), (70,100)]
# Classifiers and Hyperparameters
clfs = {
    'LDA': (LinearDiscriminantAnalysis, {'solver':['svd'], 'tol': np.logspace(-2, 2, 5)}),
    'LR': (LogisticRegression, {'C': np.logspace(-6, 3, 10), 'solver': ['liblinear', 'lbfgs'], 'penalty': ['l2']}),
    'CART': (DecisionTreeClassifier, {'criterion': ['gini'], 'ccp_alpha': np.logspace(-7, -3, 6)),
    'KNN': (KNeighborsClassifier, {'n_neighbors': np.logspace(1, 3, num=10, dtype=int), 'metric': ['euclidean']})
    'RF': (RandomForestClassifier, {'n_estimators': [200], 'max_features': ['auto', 'sqrt'], 'max_depth': [2, 5, None], 'min_samples_split': [2, 5]}),
    'SVM': (LinearSVC, {'C': np.logspace(-10, 1, 10)})
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


# Grid search
np.random.seed(42) # Reproducible result
with open('hyperparameters.txt', 'a') as f:
    f.write(f'======================={datetime.now()}============================\n')
    for SN_range in SN_ranges:
        f.write(f'***********************{SN_range}***********************\n')
        # Add noise to val set
        SNs = np.random.uniform(low=1, high=1, size=X_val.shape[0])
        # 2d array: SNs[realization][BVRI]
        SNs = np.repeat(SNs, 4, axis=0).reshape(X_val.shape[0], 4)
        X_noisy = X_val + np.random.normal(loc=0, scale=X_val/SNs)
        while np.any(X_noisy < 0): # Avoid negative flux
            X_noisy[X_noisy < 0] = X_val[X_noisy < 0] + np.random.normal(loc=0, scale=X_val/SNs)
            
        X_val = scaler.transform(X_noisy)
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
