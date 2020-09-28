#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# In[3]:


df=pd.read_csv('C:/Users/MEGHANA/Desktop/Coursera/googleplaystore.csv')


# In[4]:


df.head()


# In[5]:


df.tail()


# In[6]:


df = df[df['Content Rating'] == 'Everyone']
df = df[df['Size'].str[-1] == 'M']
df['Size'] = df['Size'].str[:-1]
grouped = df.groupby('Android Ver')
df = grouped.filter(lambda x: x['Android Ver'].count() > 500)
grouped = df.groupby('Category')
df = grouped.filter(lambda x: x['Category'].count() > 100)
df['Rating'] = df['Rating'].round()
df = df[~df['Installs'].isin(['500,000,000+','50,000,000+','1+','5+','0+','10+','100,000,000+','50+'])]
df['Installs'] = df['Installs'].str[:-1]
df = df[['App','Category','Reviews','Size','Installs','Genres','Android Ver','Rating']]
df['Reviews'] = df['Reviews'].astype(int)
df['Installs'] = df['Installs'].str.replace(',', '').astype(int)
df['Size'] = df['Size'].astype(float)
df.head()


# In[7]:


df.info()


# In[8]:


df=df.dropna()
df.info()


# In[9]:


plt.figure(figsize=(25,8))
sns.distplot(df['Size'], kde = False, color ='red', bins = 30)


# In[10]:


plt.figure(figsize=(25,8))
sns.countplot(x='Installs', data=df)


# In[11]:


plt.figure(figsize=(25,8))
sns.distplot(df['Reviews'][df['Installs'] == 5000000])


# In[12]:


plt.figure(figsize=(25,8))
sns.swarmplot(x='Rating', y='Size', data=df)


# In[13]:


plt.figure(figsize=(25,8))
# Count Plot (a.k.a. Bar Plot)
sns.countplot(x='Installs',hue='Rating', data=df)


# In[14]:


plt.figure(figsize=(25,8))
sns.boxplot(
    data=df,
    x='Rating',
    y='Size')


# In[15]:


from sklearn.preprocessing import LabelEncoder


# In[16]:


df.head()


# In[17]:


labelencoder = LabelEncoder()
df.loc[ : ,'Genres'] = labelencoder.fit_transform(df['Genres'])
df.loc[ : ,'Android Ver'] = labelencoder.fit_transform(df['Android Ver'])
df.loc[ : ,'Category'] = labelencoder.fit_transform(df['Category'])
df.head()


# In[18]:


x = df[['Category','Reviews','Size','Genres','Android Ver']]
y = df['Rating'].astype(str)


# In[19]:


y.unique()


# In[20]:


x.shape


# In[21]:


y.shape


# In[22]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.20, random_state = 5)


# In[23]:


x_train.shape, x_test.shape


# In[24]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix


# In[25]:


classifier = KNeighborsClassifier(n_neighbors = 2)


# In[26]:



classifier.fit(x_train, y_train)


# In[27]:


y_pred = classifier.predict(x_test)
print("Accuracy:",metrics.accuracy_score(y_pred,y_test))


# In[28]:


cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm, annot = True, fmt='g')


# In[29]:


accuracy = []
neighbors = list(range(1,30))
train_results = []
test_results = []
for n in neighbors:    
    knn = KNeighborsClassifier (n_neighbors=n)
    knn.fit(x_train,y_train)
    pred_i = knn.predict(x_test)
    accuracy.append(metrics.accuracy_score(pred_i,y_test))


# In[30]:


import plotly.graph_objects as go

x = list(range(1,60))

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=x,
    y=accuracy,
    mode='lines+markers'
))
fig.show()


# In[31]:


classifier = KNeighborsClassifier(n_neighbors = 21)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
print("Accuracy:",metrics.accuracy_score(y_pred,y_test))


# In[32]:


cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm, annot = True, fmt='g')


# In[33]:


from sklearn.model_selection import GridSearchCV
#making the instance
model = KNeighborsClassifier(n_jobs=-1)
#Hyper Parameters Set
params = {'n_neighbors':list(range(1,30)),
          'leaf_size':list(range(1,5)),
          'weights':['uniform', 'distance'],
          'metric':['minkowski','manhattan','euclidean'],
          'algorithm':['auto', 'ball_tree','kd_tree','brute']}
model1 = GridSearchCV(model, param_grid=params, n_jobs=1)
#Learning
model1.fit(x_train,y_train)
#The best hyper parameters set
print("Best Hyper Parameters:\n",model1.best_params_)
#Prediction
prediction=model1.predict(x_test)
print("Accuracy:",metrics.accuracy_score(prediction,y_test))
#evaluation(Confusion Metrix)
cm = confusion_matrix(prediction, y_test)
sns.heatmap(cm, annot = True, fmt='g')


# In[34]:


print(prediction[:10])
print(y_test[:10].values)


# In[ ]:




