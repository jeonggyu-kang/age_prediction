import pandas as pd

df = pd.read_csv("./ct.csv")

print (df)
pd.crosstab(df."0", df."1")