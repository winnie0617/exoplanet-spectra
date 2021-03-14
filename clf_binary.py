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
    'LR': LogisticRegression(C=100), 
    'LDA': LinearDiscriminantAnalysis(solver='svd'),
    'KNN': KNeighborsClassifier(leaf_size=10, weights='distance', n_neighbors=1),
    'CART': DecisionTreeClassifier(criterion='gini', splitter='best', ccp_alpha=3.563e-06),
    'NB': GaussianNB(),
    'SVM': LinearSVC(C=0.1),
    'RF': RandomForestClassifier(n_estimators=247, ccp_alpha=3.563e-06)
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
REALIZATION = 1000
np.random.seed(42) # Reproducible result

# print("X2", X2)

SNs = np.random.uniform(low=1, high=100, size=REALIZATION)
# 2d array: SNs[realization][BVRI]
SNs = np.repeat(SNs, 4, axis=0).reshape(REALIZATION, 4)

X2 = (X_test.var() + X_test.mean()**2).to_numpy()
# Turn into 2d array too
X2 = np.broadcast_to(X2, (REALIZATION, 4))

sigma2s = X2 / SNs

# 3d array: noises[realization][each data point][BVRI]
sigma2s = sigma2s.reshape(REALIZATION, 1, 4)
noises = np.apply_along_axis(lambda x : np.random.normal(0, x**(1/2), X_test.shape[0]), 1, sigma2s)

# df, realizations (sn) as rows, columns are the classifiers
df = pd.DataFrame(columns=list(clfs.keys()))

# Get accuracies for each realization
i = 0
for noise in noises:
    print(f'{i} realizations')
    i += 1
    X_noisy = X_test + noise
    X_scaled = scaler.transform(X_noisy)

    # Save accuracy for each model
    res = {}
    for name, clf in clfs.items():
        y_pred = clf.predict(X_scaled)
        res[name] = balanced_accuracy_score(y_pred, y_test)
    print(res)
    df = df.append(res, ignore_index=True)

df.insert(loc=0, column='S/N', value=SNs[:,0])
df.to_csv('binary_all.csv', index=False)

