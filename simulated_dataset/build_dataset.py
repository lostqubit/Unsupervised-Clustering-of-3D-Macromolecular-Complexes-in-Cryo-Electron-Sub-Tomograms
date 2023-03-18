import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

base_dir = os.path.join(os.getcwd(), 'subtomograms')

subtomograms = {'subtomogram':[],'cluster_id':[],'pdb_id':[]}
np.random.seed(0)

with open('data_config.json') as f:
    data = json.load(f)
    
np.random.shuffle(data)
for duh in data:
    subtomogram = os.path.join(base_dir,duh['subtomogram'].split('subtomograms')[1][1:])
    #print(subtomogram,duh['pdb_id'],duh['cluster_label'])
    subtomograms['subtomogram'].append(subtomogram)
    subtomograms['cluster_id'].append(duh['cluster_label'])
    subtomograms['pdb_id'].append(duh['pdb_id'])
    
df = pd.DataFrame(subtomograms)

df_train,df_test = train_test_split(df,random_state=0,test_size = 0.1)

df_train.to_csv('train.csv',index=False)
df_test.to_csv('test.csv',index=False)