#%%
import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
# %%
df = pd.read_csv('loan_data.csv')
df.head()
# %%
df.info()
# %%
df.describe()
# %%

df[df['credit.policy'] == 1]['fico'].hist(bins=30,label='credit.policy=1')
df[df['credit.policy'] == 0]['fico'].hist(bins=30,label='credit.policy=0')
plt.legend()
plt.xlabel('FICO')
# %%
df[df['not.fully.paid'] == 0]['fico'].hist(bins=30,label='not.fully.paid=0')
df[df['not.fully.paid'] == 1]['fico'].hist(bins=30,label='not.fully.paid=1')
plt.legend()
plt.xlabel('FICO')
# %%
plt.figure(figsize=(11,7))
sns.countplot(x='purpose',hue=df['not.fully.paid'],data=df)
# %%
sns.jointplot(x='fico',y='int.rate',data=df)
# %%
plt.figure(figsize=(11,7))
sns.lmplot(y='int.rate',x='fico',data=df,hue='credit.policy',
           col='not.fully.paid',palette='Set1')
# %%
cat_feats = ['purpose']

final_data = pd.get_dummies(df,columns=cat_feats,drop_first=True)
final_data.info()
# %%
final_data.head()
# %%
from sklearn.model_selection import train_test_split

X = final_data.drop('not.fully.paid',axis=1)
y = final_data['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
# %%
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()

dtree.fit(X_train,y_train)
# %%
dtree_predictions = dtree.predict(X_test)

# %%
from sklearn.metrics import confusion_matrix,classification_report

print(confusion_matrix(y_test,dtree_predictions))
print('\n')
print(classification_report(y_test,dtree_predictions))
# %%
## do the same test but using random forest to check if get any improvements
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()

rfc.fit(X_train,y_train)
# %%
rfc_predict = rfc.predict(X_test)
# %%
print(confusion_matrix(y_test,rfc_predict))
print('\n')
print(classification_report(y_test,rfc_predict))

# %%
 ## as comparasion between the two models we can see an improvement on the detection of the clients that didn`t fully paid theirs borrow