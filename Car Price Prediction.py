#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df=pd.read_csv('car details.csv')


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


print(df.seller_type.unique())
print(df.transmission.unique())
print(df.owner.unique())
print(df.fuel.unique())


# In[6]:


df.isnull().sum()


# In[7]:


df.describe()


# In[8]:


df.columns


# In[9]:


final_ds = df[['year', 'selling_price', 'km_driven', 'fuel', 'seller_type',
       'transmission', 'owner']]


# In[10]:


final_ds.head()


# In[11]:


final_ds['Current_year'] = 2023


# In[12]:


final_ds['no_year'] = final_ds.Current_year - final_ds.year


# In[13]:


final_ds.drop(['year'],axis=1,inplace=True)


# In[14]:


final_ds.drop(['Current_year'],axis=1,inplace=True)


# In[15]:


final_ds.head()


# In[16]:


final_ds = pd.get_dummies(final_ds,drop_first=True)


# In[17]:


final_ds.head()


# In[18]:


final_ds.corr()


# In[19]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[20]:


corrmat = final_ds.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(15,15))
g = sns.heatmap(final_ds[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[21]:


final_ds.head()


# In[22]:


X = final_ds.iloc[:,1:]
y = final_ds.iloc[:,0]


# In[23]:


X.head()


# In[24]:


y.head()


# In[25]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import ExtraTreesRegressor


# In[26]:


models = []

models.append(('Linear Regression', LinearRegression()))
models.append(('Decision Tree', DecisionTreeRegressor()))
models.append(('Random Forest', RandomForestRegressor()))
models.append(('Support Vector Regression', SVR()))
models.append(('Gradient Boosting Regression', GradientBoostingRegressor()))
models.append(('K-Nearest Neighbors', KNeighborsRegressor()))
models.append(('Extra Trees', ExtraTreesRegressor()))


# In[27]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)


# In[28]:


from sklearn.metrics import r2_score
r2_scores_initial = []

for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2_sco = r2_score(y_test, y_pred)
    r2_scores_initial.append(r2_sco)
    print(name, 'r2_score:', r2_sco)


# In[29]:


from sklearn.model_selection import cross_val_score, KFold

k_folds = KFold(n_splits=5, shuffle=True, random_state=26)
r2_scores_final = []

for name, model in models:
    accuracy_scores = cross_val_score(model, X_train, y_train, cv=k_folds, scoring='r2')
    score = accuracy_scores.mean()
    r2_scores_final.append(score)
    print(name, 'mean r2_score:', score)


# In[30]:


for i in range(len(models)):
    if r2_scores_final[i] > r2_scores_initial[i]:
        print(f"{models[i][0]}: r2 score improved from {r2_scores_initial[i]} to {r2_scores_final[i]}")


# In[34]:


best_model_index_1 = r2_scores_final.index(max(r2_scores_final))
best_model_index_2 = r2_scores_initial.index(max(r2_scores_initial))
if(r2_scores_final[best_model_index_1] > r2_scores_initial[best_model_index_2]):
    best_model_index = best_model_index_1
    best_r2_score = r2_scores_final[best_model_index_1]
else: 
    best_model_index = best_model_index_2
    best_r2_score = r2_scores_initial[best_model_index_2]

print("The best model is",models[best_model_index][0],"with r2_score =",best_r2_score)

best_model = models[best_model_index][1]
best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)


# In[35]:


fig, axes = plt.subplots(ncols=2, figsize=(10, 5))

# plot the distribution of the residuals using a distplot on the left side
sns.histplot(y_test - y_pred, kde=False, ax=axes[0])
axes[0].set_title('Distribution of Residuals')
axes[0].set_xlabel('Residuals')
axes[0].set_ylabel('Count')

# plot the relationship between the predicted and actual target values using a scatter plot on the right side
axes[1].scatter(y_test, y_pred)
axes[1].set_title(f'Predicted vs Actual for {models[best_model_index][0]}')
axes[1].set_xlabel('Actual values')
axes[1].set_ylabel('Predicted values')

plt.show()


# In[36]:


import pickle
file = open('model.pkl', 'wb')

pickle.dump(best_model, file)

