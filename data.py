import pandas as pd
from sklearn.model_selection import train_test_split

baltimore=pd.read_csv('BPD_Part_1_Victim_Based_Crime_Data.csv')
baltimore["Weapon"].fillna("None", inplace=True)
baltimore.dropna(inplace=True)

baltimore['Post'] = baltimore['Post'] /baltimore['Post'].abs().max()
baltimore['Location']=baltimore['Location'].str.lower()
baltimore['Description']=baltimore['Description'].str.lower()
baltimore['Weapon']=baltimore['Weapon'].str.lower()
baltimore['Premise']=baltimore['Premise'].str.lower()
baltimore['District']=baltimore['District'].str.lower()
baltimore['CrimeCode']=baltimore['CrimeCode'].str.lower()
baltimore['Neighborhood']=baltimore['Neighborhood'].str.lower()
baltimore['Inside/Outside']=baltimore['Inside/Outside'].str.lower()


baltimore_train, baltimore_test = train_test_split(baltimore, test_size=0.1, random_state=1)
baltimore_train, baltimore_dev= train_test_split(baltimore_train, test_size=0.25, random_state=1)

baltimore_test.to_csv("baltimore_test.csv", encoding="utf-8", index=False)
baltimore_dev.to_csv("baltimore_dev.csv", encoding="utf-8", index=False)
baltimore_train.to_csv("baltimore_train.csv", encoding="utf-8", index=False)