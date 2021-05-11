import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from util import *

# Classifiers and Hyperparameters
def get_clfs(names):
    first_clfs = {
        'LR': LogisticRegression(C=1, penalty='l2', solver='liblinear'), 
        'LDA': LinearDiscriminantAnalysis(solver='svd', tol=0.01),
        'KNN': KNeighborsClassifier(weights='uniform', n_neighbors=201),
        'CART': DecisionTreeClassifier(criterion='gini', splitter='best', ccp_alpha=2e-5),
        'NB': GaussianNB(),
        'SVM': LinearSVC(C=10, dual=False),
        'RF': RandomForestClassifier(n_estimators=200, min_samples_split=50, ccp_alpha=0.0003)
    }
    # Add the two voting classifiers
    clfs = first_clfs.copy()
    clfs['MVH'] = VotingClassifier(estimators=list(first_clfs.items()), voting='hard')
    # Remove SVM for MVS
    del first_clfs['SVM']
    clfs['MVS'] = VotingClassifier(estimators=list(first_clfs.items()), voting='soft')

    # Get binary label
    data = pd.read_pickle('full_colors.pkl')
    data['Y'] = data.biota_percentage != 0
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(data[['B','V','R','I']], data['Y'], test_size=0.2, random_state=42)

    # Add noise and aggregate negative samples
    X_train, y_train = upsample_minority(X_train, y_train)
    X_train = add_noise(X_train)

    # Scale data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)

    # Train and save model
    return_clfs = {}
    for name, clf in clfs.items():
        if name in names:
            print(name)
            clf.fit(X_train, y_train)
            return_clfs[name] = clf
    
    return return_clfs