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

from sklearn.inspection import plot_partial_dependence
from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt

# Classifiers and Hyperparameters
first_clfs = {
    'LR': LogisticRegression(C=100), 
    'LDA': LinearDiscriminantAnalysis(solver='svd'),
    'KNN': KNeighborsClassifier(leaf_size=10, weights='distance', n_neighbors=1),
    # 'CART': DecisionTreeClassifier(criterion='gini', splitter='best', ccp_alpha=3.563e-06),
    # 'NB': GaussianNB(),
    # 'SVM': LinearSVC(C=0.1),
    # 'RF': RandomForestClassifier(n_estimators=247, ccp_alpha=3.563e-06)
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

# # Train and save model
# for name, clf in clfs.items():
#     print(name)
#     clf.fit(X_train, y_train)

# # Add Gaussian noise
# REALIZATION = 1000
# np.random.seed(42) # Reproducible result

# df = pd.DataFrame(columns=list(clfs.keys()))

# # Get accuracies for each realization
# SNs = list(range(1, 2))
# for SN in SNs:
#     print(f'SNR: {SN}')
#     X_noisy = np.copy(X_test)
#     for noise_realization in range(REALIZATION):
#         X_noisy += np.random.normal(loc=0, scale=X_test/SN/(REALIZATION**(1/2))) # Here already using noisy data for sd??
#         X_noisy[X_noisy < 0] = 0 # Avoid negative flux
#     X_scaled = scaler.transform(X_noisy)

# # Plot partial dependence
# for name, clf in clfs.items():
#     print(np.max(X_train))
#     plot_partial_dependence(clf, X_train, [0,1,2,3])
#     plt.show()

# param_range = np.arange(1,10,2)
# train_scores, test_scores = validation_curve(KNeighborsClassifier(), X_train, y_train, 'n_neighbors', param_range, cv=5)
param_range = np.logspace(-2, 2, 5)
train_scores, test_scores = validation_curve(LinearSVC(), X_train, y_train, 'C', param_range, cv=5)
print(train_scores, test_scores)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("Validation Curve with SVM")
plt.xlabel(r"$\gamma$")
plt.ylabel("Score")
plt.ylim(0.0, 1.1)
lw = 2
plt.semilogx(param_range, train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
             color="navy", lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")
plt.show()
