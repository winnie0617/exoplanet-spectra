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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

from util import *

# Classifiers and Hyperparameters
first_clfs = {
    'LR': LogisticRegression(C=1, penalty='l2', solver='liblinear'), 
    'LDA': LinearDiscriminantAnalysis(solver='svd', tol=0.01),
    'KNN': KNeighborsClassifier(weights='uniform', n_neighbors=201),
    'CART': DecisionTreeClassifier(criterion='gini', splitter='best', ccp_alpha=2e-5),
    'NB': GaussianNB(),
    'SVM': LinearSVC(C=10),
    'RF': RandomForestClassifier(n_estimators=200, min_samples_split=50, ccp_alpha=0.0003)
}
# Add the two voting classifiers
clfs = {}
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
for name, clf in clfs.items():
    print(name)
    clf.fit(X_train, y_train)

# ========================================== Test Set ================================================

# Add Gaussian noise
REALIZATION = 100
np.random.seed(42) # Reproducible result

cols = ['label']
cols.extend(list(clfs.keys()))
df = pd.DataFrame(columns=cols)

# Get accuracies for each realization
# SNs = list(range(1, 101))

SNs = [10, 100]
fig, ax = plt.subplots(len(clfs), len(SNs))
fig.text(0.5, 0.04, 'Predicted label', ha='center')
fig.text(0.04, 0.5, 'True label', va='center', rotation='vertical')
r = c = 0
for SN in SNs:
    if r == 0: # setting suptitle
        ax[r,c].set_title(f'S/N={SN}')
    print(f'SNR: {SN}')
    X_noisy = np.empty((0, X_test.shape[1]))
    for noise_realization in range(REALIZATION):
        X_noisy = np.append(X_noisy, add_noise(X_test.to_numpy(), SNs=SN), axis=0)
    X_scaled = scaler.transform(X_noisy)
    y_true = np.tile(y_test, REALIZATION)
    print("y_test", y_test, "y_true", y_true)
    # Save all predictions for each model
    res = {'S/N': SN}
    for name, clf in clfs.items():
        y_pred = clf.predict(X_scaled)
        cf_matrix = confusion_matrix(y_true, y_pred)
        normalize = lambda x: x/np.sum(x)
        cf_matrix = np.apply_along_axis(normalize, 1, cf_matrix)
        sns.heatmap(cf_matrix, ax=ax[r,c], vmin=0, vmax=1, annot=True, cmap='Oranges', cbar = c==0, cbar_ax=None if c else fig.add_axes([.91, .3, .03, .4]), xticklabels=['False', 'True'], yticklabels=['False', 'True'])

        if c == 0: # setting row header
            ax[r,c].annotate(name, xy=(-0.2, 0.5), xycoords='axes fraction', size='large')
        r += 1
    c += 1
    r = 0

# plt.savefig('confusion.png')
plt.show()

# df.insert(loc=0, column='S/N', value=SNs)
# df.to_csv('binary_confusion.csv', index=False)

