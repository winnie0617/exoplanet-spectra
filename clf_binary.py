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

# Classifiers and Hyperparameters
first_clfs = {
    'LR': LogisticRegression(C=1e-7, penalty='l2', solver='liblinear'), 
    'LDA': LinearDiscriminantAnalysis(solver='svd'),
    'KNN': KNeighborsClassifier(weights='distance', n_neighbors=300),
    'CART': DecisionTreeClassifier(criterion='gini', splitter='best', ccp_alpha=3.563e-06),
    'NB': GaussianNB(),
    'SVM': LinearSVC(C=1e-8),
    'RF': RandomForestClassifier(n_estimators=400, max_features='sqrt', max_depth=5)
}
# Add the two voting classifiers
clfs = first_clfs.copy()
clfs['MVH'] = VotingClassifier(estimators=list(first_clfs.items()), voting='hard')
# Remove SVM for MVS
del first_clfs['SVM']
clfs['MVS'] = VotingClassifier(estimators=list(first_clfs.items()), voting='soft')

# Get binary label
df = pd.read_pickle('full_colors.pkl')
df['Y'] = df.biota_percentage != 0
# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(df[['B','V','R','I']], df['Y'], test_size=0.2, random_state=42)

# Scale data
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)

# Train and save model
for name, clf in clfs.items():
    print(name)
    clf.fit(X_train, y_train)

# ========================================== Test Set ================================================

# Add Gaussian noise
REALIZATION = 100
np.random.seed(42) # Reproducible result

cols = ['S/N']
cols.extend(list(clfs.keys()))
df = pd.DataFrame(columns=cols)
print(df)

# Get accuracies for each realization
# SNs = list(range(1, 101))
SNs = [1, 3, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]
for SN in SNs:
    print(f'SNR: {SN}')
    for noise_realization in range(REALIZATION):
        X_noisy = X_test + np.random.normal(loc=0, scale=X_test/SN)
        
        while np.any(X_noisy < 0): # Avoid negative flux
            X_noisy[X_noisy < 0] = X_test[X_noisy < 0] + np.random.normal(loc=0, scale=X_test/SN)
        X_scaled = scaler.transform(X_noisy)

        # Save accuracy for each model
        res = {'S/N': SN}
        for name, clf in clfs.items():
            y_pred = clf.predict(X_scaled)
            res[name] = balanced_accuracy_score(y_test, y_pred)
        print(res)
        df = df.append(res, ignore_index=True)

# df.insert(loc=0, column='S/N', value=SNs)
df.to_csv('binary_all.csv', index=False)

