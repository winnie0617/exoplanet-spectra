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
clfs = {
    # 'LR': LogisticRegression(C=0.1, penalty='l2', solver='liblinear'), 
    # 'LDA': LinearDiscriminantAnalysis(solver='svd', tol=0.01),
    'KNN': KNeighborsClassifier(weights='distance', n_neighbors=41),
    'CART': DecisionTreeClassifier(criterion='gini', splitter='best', ccp_alpha=5e-5),
    # 'NB': GaussianNB(),
    # 'SVM': LinearSVC(C=100),
    'RF': RandomForestClassifier(n_estimators=200, min_samples_split=50, class_weight='balanced', ccp_alpha=0.0003)
}

# Get binary label
data = pd.read_pickle('full_colors.pkl')
data['Y'] = data.biota_percentage != 0
# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(data[['B','V','R','I']], data['Y'], test_size=0.2, random_state=42)
biota_types = data.biota_type.unique()

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

cols = ['S/N', ]
cols.extend(list(clfs.keys()))
df = pd.DataFrame(columns=cols)

# Get accuracies for each realization
# SNs = list(range(1, 101))
SNs = [1, 3, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]
for SN in SNs:
    print(f'SNR: {SN}')
    for noise_realization in range(REALIZATION):
        X_noisy = X_test + np.random.normal(loc=0, scale=X_test/SN)
        
        while np.any(X_noisy < 0): # Avoid negative flux
            X_noisy[X_noisy < 0] = X_test[X_noisy < 0] + np.random.normal(loc=0, scale=X_test/SN)

        X_scaled = pd.DataFrame(scaler.transform(X_noisy), index=X_noisy.index, columns=['B','V','R','I'])
        X_scaled['biota_type'] = data.biota_type[X_scaled.index]

        # Loop through each biota type
        for biota in biota_types:
            print(biota)
            # idx = data[data.biota_type==biota].index.intersection(X_test.index)
            # print("idx", idx)
            X_test_sub = X_scaled[X_scaled.biota_type==biota].drop(labels='biota_type', axis=1)
            y_test_sub = y_test[X_scaled.biota_type==biota]
            # Save accuracy for each model
            res = {'S/N': SN, 'biota': biota}
            for name, clf in clfs.items():
                y_pred = clf.predict(X_test_sub)
                res[name] = balanced_accuracy_score(y_test_sub, y_pred)
            print(res)
            df = df.append(res, ignore_index=True)

df.to_csv('by_biota.csv', index=False)

