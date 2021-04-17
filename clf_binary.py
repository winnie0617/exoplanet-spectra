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

from util import add_noise

# Classifiers and Hyperparameters
first_clfs = {
    'LR': LogisticRegression(C=1e-10, penalty='l2', solver='liblinear'), 
    'LDA': LinearDiscriminantAnalysis(solver='svd', tol=0.01),
    'KNN': KNeighborsClassifier(weights='distance', n_neighbors=3),
    'CART': DecisionTreeClassifier(criterion='gini', splitter='best', ccp_alpha=1e-6, min_samples_split=5),
    'NB': GaussianNB(),
    'SVM': LinearSVC(C=100),
    'RF': RandomForestClassifier(n_estimators=200, min_samples_split=50, class_weight='balanced', ccp_alpha=0.0003)
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
num_pos = y_train.sum()
factor = num_pos // (len(y_train)-num_pos)

X_train_neg = X_train[y_train == False].to_numpy()
X_train_upsampled = np.append(X_train.to_numpy(), np.repeat(X_train_neg, factor-1, axis=0), axis=0)
y_train_upsampled = np.append(y_train.to_numpy(), np.repeat(False, len(X_train_neg)*(factor-1)))
n = len(y_train_upsampled)
idx = np.arange(n)
np.random.shuffle(idx)
X_train = X_train_upsampled[idx,:]
y_train = y_train_upsampled[idx]

X_train = add_noise(X_train)

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

# Get accuracies for each realization
# SNs = list(range(1, 101))
SNs = [1, 3, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]
for SN in SNs:
    print(f'SNR: {SN}')
    for noise_realization in range(REALIZATION):
        X_noisy = add_noise(X_test, SNs=SN)
        X_scaled = scaler.transform(X_noisy)

        # Save accuracy for each model
        res = {'S/N': SN}
        for name, clf in clfs.items():
            y_pred = clf.predict(X_scaled)
            res[name] = balanced_accuracy_score(y_test, y_pred)
        print(res)
        df = df.append(res, ignore_index=True)

# df.insert(loc=0, column='S/N', value=SNs)
df.to_csv('binary_noisy_train.csv', index=False)

