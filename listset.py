import os
import pandas as pd
import random

source_file = "../project_classifiers/data/age.csv"
dest_dir = "./data"
train_ratio = 70



df = pd.read_csv(source_file)

df = df.loc[:, ['f', 'a', 's']]
df.dropna(inplace = True)

#df.rename(columns = {'input_file' : 'dcm_path' , 'age' : 'age', 'sex' : 'sex' }, inplace = True)
df.rename(columns = {'f' : 'dcm_path'  }, inplace = True)
training = []
for i in range (len(df)):
    num = random.randint(1,1000001)
    training.append(num)
   

    
df['training'] = training

df.sort_values(by=['training'], axis=0)

row_cut = int(len(df)*train_ratio/100)

# df.reset_index()

df = df.loc[:, ['dcm_path', 'a', 's']]

df_training = df.iloc[:row_cut,:]
df_test     = df.iloc[row_cut:,:]


df_training.to_parquet(dest_dir+'/train_dataset.parquet')
df_test.to_parquet(dest_dir+'/test_dataset.parquet')