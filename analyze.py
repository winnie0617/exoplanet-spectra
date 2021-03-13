#%%
import pandas as pd
import seaborn as sns
import numpy as np

df = pd.read_csv('binary_all.csv')
# df.insert(0, 'group', np.round(df.index)) 
df['S/N'] = np.round(df['S/N'])
df = pd.melt(df, id_vars=['S/N'], var_name='model', value_name='balanced_accuracy')
# df.sort_values('S/N', inplace=True)

sns.set()
sns.lineplot(data=df, x='S/N', y='balanced_accuracy', hue='model')
# mean = df.groupby(pd.cut(df.index, bins=100)).mean()
# sd = df.groupby(pd.cut(df.index, bins=100)).std()
# mean.index = mean.index.mid()
# mean
# sns.lineplot(data=mean, ci='sd')
# %%
