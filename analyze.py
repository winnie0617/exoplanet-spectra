#%%
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('binary_old_param.csv')
# df['S/N'] = np.round(df['S/N'], decimals=-1)
df = df[['S/N', 'LR', 'LDA', 'KNN', 'MVH', 'SVM', 'NB', 'CART', 'RF', 'MVS']]
print("df", df)
df = pd.melt(df, id_vars=['S/N'], var_name='model', value_name='balanced_accuracy')

sns.set()
fig, ax = plt.subplots()
plot = sns.lineplot(data=df, x='S/N', y='balanced_accuracy', hue='model', ax=ax)
ax.set_ylim(0.5, 0.77)
plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
# plot.get_figure().savefig('binary_updated.png')

# %% 
