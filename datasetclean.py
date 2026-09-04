import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

encoder=LabelEncoder()

df1=pd.read_csv("C:/Users/mahij/OneDrive/Desktop/mjsem4/aiproject/eng.csv",encoding='cp1252')
df1.rename(columns={'Column1':'Target','Column2':'Text'},inplace=True)
df1['Target']=encoder.fit_transform(df1['Target'])
df1.drop_duplicates(inplace=True)
df1.dropna(inplace=True)
print(df1.head())


df2 = pd.read_csv(r"C:\Users\mahij\OneDrive\Desktop\mjsem4\aiproject\spam_hindi.csv", encoding="utf-8-sig")
df2.rename(columns={'v1':'Target','v2':'Text'},inplace=True)
df2['Target']=encoder.fit_transform(df2['Target'])
df2.drop_duplicates(inplace=True)
df2.dropna(inplace=True)
print(df2.head())

df3 = pd.read_csv(r"C:\Users\mahij\OneDrive\Desktop\mjsem4\aiproject\mail_data.csv", encoding="utf-8-sig")
df3.rename(columns={'Category':'Target','Message':'Text'},inplace=True)
df3['Target']=encoder.fit_transform(df3['Target'])
df3.drop_duplicates(inplace=True)
df3.dropna(inplace=True)
print(df3.head())


df4= pd.concat([df1, df2,df3])
df4= shuffle(df4,random_state=42).reset_index(drop=True)
df4.to_csv(r"C:\Users\mahij\OneDrive\Desktop\mjsem4\aiproject\combined_dataset.csv", 
           index=False, 
           encoding='utf-8-sig')
print("Data preprocessed!")