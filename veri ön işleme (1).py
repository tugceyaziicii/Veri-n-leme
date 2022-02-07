#!/usr/bin/env python
# coding: utf-8

# In[98]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats 
veri=pd.read_csv("Hitters.csv")


# In[99]:


veri.head()


# In[100]:



veri.tail()


# In[101]:



veri.info() # 20 adet değişken bulunmakta
#kategorik değişkenler league,division,newleague
#nümerik değişkenler geriye kalanlar
#salary kısmında eksik gözlem var


# In[102]:



veri.describe().T


# In[103]:


veri.isnull().sum() #salary kısmında 59 tane eksik gözlem var


# In[126]:



from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
salary=veri.iloc[:,-2:-1].values
print(salary)


# In[132]:


imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
imputer=imputer.fit(veri[["Salary"]])

veri["Salary"]=imputer.transform(veri[["Salary"]])
print(veri)


# In[109]:


#kategorik verileri nümerik verilere dönüştürme-object değerleri sayısal değere dönüştür
league=veri.iloc[:,13:14].values
division = veri.iloc[:,14:15].values
newLeague = veri.iloc[:,-1:].values  
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

veri["League"]=le.fit_transform(veri["League"])
veri["Division"]=le.fit_transform(veri["Division"])
veri["NewLeague"]=le.fit_transform(veri["NewLeague"])


# In[ ]:




