import os
import pandas as pd
import random

source_file = "./cxr_clinical.csv"
dest_dir = "./data"
input_file = "file_name"
outcome = "age"
train_ratio = 70


df = pd.read_csv(source_file)

df = df.loc[:, [input_file, outcome]]
df.dropna(inplace = True)

df.rename(columns = {input_file : 'dcm_path' , outcome : 'score' }, inplace = True)

training = []
for i in range (len(df)):
    num = random.randint(1,1000001)
    training.append(num)
   

    
df['training'] = training

df.sort_values(by=['training'], axis=0)

row_cut = int(len(df)*train_ratio/100)

# df.reset_index()

df = df.loc[:, ['dcm_path', 'score']]

df_training = df.iloc[:row_cut,:]
df_test     = df.iloc[row_cut:,:]


df_training.to_parquet(dest_dir+'/train_dataset.parquet')
df_test.to_parquet(dest_dir+'/test_dataset.parquet')