#%%
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

#%%
df = pd.read_csv('binary_upsampled.csv')
print(df)
# df['S/N'] = np.round(df['S/N'], decimals=-1)
df = df[['S/N', 'LR', 'LDA', 'KNN', 'MVH', 'SVM', 'NB', 'CART', 'RF', 'MVS']]
df = pd.melt(df, id_vars=['S/N'], var_name='model', value_name='balanced_accuracy')
print("df", df)

sns.set()
fig, ax = plt.subplots()
plot = sns.lineplot(data=df, x='S/N', y='balanced_accuracy', hue='model', ax=ax, err_style='band', ci=99)
ax.set_ylim(0.5, 0.77)
plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
# plot.get_figure().savefig('binary_updated.png')

# %% By biota plot
df = pd.read_csv('by_biota.csv')

sns.set()
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8,12), sharex=True)
plot = sns.lineplot(data=df, x='S/N', y='KNN', hue='biota', ax=ax1)
plot = sns.lineplot(data=df, x='S/N', y='CART', hue='biota', ax=ax2, legend=False)
plot = sns.lineplot(data=df, x='S/N', y='RF', hue='biota', ax=ax3, legend=False)
ax1.set_ylim(0.5, 0.89)
ax2.set_ylim(0.5, 0.89)
ax3.set_ylim(0.5, 0.89)
# plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
# %% Visualize dataset
data = pd.read_pickle('full_colors.pkl')
print(data)
data['Y'] = data.biota_percentage != 0
# sns.displot(x=data['B'], y=data['biota_percentage'], kind='kde')
plt.plot(data['V'], data['biota_percentage'], '.')
#,'V','R','I'
# %% Verifying sampling
from util import *
import pandas as pd
from sklearn.model_selection import train_test_split
# Get binary label
data = pd.read_pickle('full_colors.pkl')
data['Y'] = data.biota_percentage != 0
# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(data[['B','V','R','I']], data['Y'], test_size=0.2, random_state=42)

# Add noise and aggregate negative samples
X_train, y_train = upsample_minority(X_train, y_train)
u = np.unique(X_train, axis=0)
# X_train = add_noise(X_train)

# # Scale data
# scaler = StandardScaler()
# scaler.fit(X_train)
# X_train = scaler.transform(X_train)



# %%
