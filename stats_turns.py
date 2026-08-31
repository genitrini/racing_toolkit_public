import pandas as pd
import numpy as np
import os
import seaborn as sns
from utilities import *

data = pd.read_pickle('df_allpoints.pkl')
data.insert(9, 'valnorm', np.round((data['value'] % 10000) / 100, 2))

id = 227
track = 'brandshatch'

datasubj = data[(data.subject == id) & (data.track == track)].reset_index(drop = True)
datasubj['quantity'] = datasubj['quantity'].astype('category')
#datasubj2 = datasubj.groupby(['subject', 'track', 'session', 'block', 'turn',
                             # 'quantity'])['valnorm'].mean().reset_index()
plt.figure()
sns.boxplot(datasubj, x='turn', y='valnorm', hue = 'quantity')
plt.xlabel('Turn #', fontsize = 30)
plt.ylabel("Location wrt lap completion [%]", fontsize = 30)
plt.title('Distribution of points in ' + str(datasubj.track[0]) + ': ID '
          + str(datasubj.subject[0]) + ' ' + str(datasubj.level[0]), fontsize = 40)
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)
plt.legend(fontsize = 20)
plt.show()

print("done")

