# Diabetes database

import pandas as pd
dbdf = pd.read_csv('/content/Diabetes Database.csv')

dbdf.head()

print('Different attributes provided in the dataset to find which patients have diabeties are...')
print(dbdf.columns.values)

# Here, Outcome value 0 means non-diabetic and 1 means diabetic
# Class distribution
dbdf.groupby(dbdf.columns[-1]).size()

# Data distribution
import matplotlib.pyplot as plt

fig = plt.figure(figsize = (9,9))
ax = fig.gca()
dbdf.hist(ax = ax)

plt.show()

dbdf.groupby(dbdf.columns[-1]).hist(figsize = (9,9))
print('OUTCOME - 0 (is 1st subplot) & OUTCOME - 1 (is 2nd subplot)')
plt.show()

# Finding whether any missing values are present in the data
dbdf.isna().sum()

# Finding whether any null values are present in the data
dbdf.isnull().sum()

# For diabetes dataset, possible outliers would be attributes value equals to zero. 
# So we would check whether attribute values are zero.
for i in range(1, len(dbdf.columns) - 3):
  if(len(dbdf.columns[i]) > 12):
    print('Total no. of rows having 0 in ',dbdf.columns[i],' is ',dbdf[dbdf[dbdf.columns[i]] == 0].shape[0])
  elif(len(dbdf.columns[i]) > 5):
    print('Total no. of rows having 0 in ',dbdf.columns[i],' is\t ',dbdf[dbdf[dbdf.columns[i]] == 0].shape[0])
  else:
    print('Total no. of rows having 0 in ',dbdf.columns[i],' is\t\t ',dbdf[dbdf[dbdf.columns[i]] == 0].shape[0])

# Since for SkinThickness and Insulin, Total values is large.
# It will not be a better option to remove those rows.
# So we try to remove the rows of atrribute Glucose, BloodPressure and BMI

mdf = dbdf[(dbdf.Glucose != 0) & (dbdf.BloodPressure != 0) & (dbdf.BMI != 0)]

X = mdf.iloc[:,:-1].values
y = mdf.iloc[:,-1].values.reshape(-1,1)

# Split the data into train and test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Model selection

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

models = []
models.append(('KNN', KNeighborsClassifier()))
models.append(('LR', LogisticRegression()))
models.append(('DT', DecisionTreeClassifier()))
models.append(('GNB', GaussianNB()))
models.append(('RF', RandomForestClassifier()))

from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")


names = []
scores = []

for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    scores.append(accuracy_score(y_test, y_pred))
    names.append(name)

tr_split = pd.DataFrame({'Name': names, 'Score': scores})
print(tr_split)

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

strat_k_fold = StratifiedKFold(n_splits=10, random_state=10)

names = []
scores = []

for name, model in models:
    
    score = cross_val_score(model, X, y, cv=strat_k_fold, scoring='accuracy').mean()
    names.append(name)
    scores.append(score)

kf_cross_val = pd.DataFrame({'Name': names, 'Score': scores})
print(kf_cross_val)

import seaborn as sns

axis = sns.barplot(x = 'Name', y = 'Score', data = kf_cross_val)
axis.set(xlabel='Classifier', ylabel='Accuracy')

for p in axis.patches:
    height = p.get_height()
    axis.text(p.get_x() + p.get_width()/2, height + 0.005, '{:1.4f}'.format(height), ha="center") 
    
plt.show()
