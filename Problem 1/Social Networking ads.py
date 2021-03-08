# Social networking ads

import pandas as pd
import numpy as np
import math

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import warnings
warnings.filterwarnings("ignore")

# Data exploration
snadf = pd.read_excel('/content/Social_Network_Ads.xlsx')
snadf.head()

print('Different attributes provided in the dataset to find whether people purchase the items by seeing social network ads are...')
print(snadf.columns.values)

# Here, Outcome value 0 means not purchased and 1 means purchased
# Class distribution
snadf.groupby(snadf.columns[-1]).size()

# Data distribution
fig = plt.figure(figsize = (6,6))
ax = fig.gca()
snadf.hist(ax = ax)

plt.show()

snadf.groupby(snadf.columns[-1]).hist(figsize = (6,6))
print('OUTCOME - 0 (is 1st subplot) & OUTCOME - 1 (is 2nd subplot)')

plt.show()

# Finding whether any missing values are present in the data
snadf.isna().sum()

# Finding whether any null values are present in the data
snadf.isnull().sum()

# Outliers
# For social network ads dataset, possible outliers would be attributes value equals to zero. 
# So we would check whether attribute values are zero.
for i in range(2, len(snadf.columns) - 1):
  if(len(snadf.columns[i]) > 12):
    print('Total no. of rows having 0 in ',snadf.columns[i],'is',snadf[snadf[snadf.columns[i]] == 0].shape[0])
  elif(len(snadf.columns[i]) > 5):
    print('Total no. of rows having 0 in ',snadf.columns[i],'is\t  ',snadf[snadf[snadf.columns[i]] == 0].shape[0])
  else:
    print('Total no. of rows having 0 in ',snadf.columns[i],'is\t\t  ',snadf[snadf[snadf.columns[i]] == 0].shape[0])

# Encoding for gender attribute
le = LabelEncoder()
snadf[snadf.columns[1]] = le.fit_transform(snadf[snadf.columns[1]])
print(snadf.columns[1])
snadf.head()

# Since no attributes have value 0, so there are no outliers in our data.
X = snadf.iloc[:,[2,3]].values
y = snadf.iloc[:,-1].values.reshape(-1,1)

# Split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Model selection
models = []
models.append(('KNN', KNeighborsClassifier()))
models.append(('LR', LogisticRegression()))
models.append(('DT', DecisionTreeClassifier()))
models.append(('GNB', GaussianNB()))
models.append(('RF', RandomForestClassifier()))

# Finding the accuracy of 5 different models
names = []
scores = []

for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    scores.append(accuracy_score(y_test, y_pred))
    names.append(name)

tr_split = pd.DataFrame({'Name': names, 'Score': scores})
print(tr_split)

# K-Fold cross validation
strat_k_fold = StratifiedKFold(n_splits=10, random_state=0)

names = []
scores = []

for name, model in models:
    
    score = cross_val_score(model, X, y, cv=strat_k_fold, scoring='accuracy').mean()
    names.append(name)
    scores.append(score)

kf_cross_val = pd.DataFrame({'Name': names, 'Score': scores})
print(kf_cross_val)

axis = sns.barplot(x = 'Name', y = 'Score', data = kf_cross_val)
axis.set(xlabel='Classifier', ylabel='Accuracy')

for p in axis.patches:
    height = p.get_height()
    axis.text(p.get_x() + p.get_width()/2, height + 0.005, '{:1.4f}'.format(height), ha="center") 
    
plt.show()

# Random forest model shows good accuracy when compared to other models.
# So we first try to plot the graph for Random forest using inbuilt function, 
# and then we'll try to plot graph for Random forest implemented from scratch.

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Random Forest model(Inbuilt function)
rfc = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print('Confusion matrix\n(',cm[0][0],'\t',cm[0][1],' )\n(',cm[1][0],'\t',cm[1][1],')')


# Random forest model from scratch
class RandomForest():
    def __init__(self, x, y, n_trees, n_features, sample_sz, depth=10, min_leaf=5):
        np.random.seed(12)
        if n_features == 'sqrt':
            self.n_features = int(np.sqrt(x.shape[1]))
        elif n_features == 'log2':
            self.n_features = int(np.log2(x.shape[1]))
        else:
            self.n_features = n_features
        self.x, self.y, self.sample_sz, self.depth, self.min_leaf  = x, y, sample_sz, depth, min_leaf
        self.trees = [self.create_tree() for i in range(n_trees)]

    def create_tree(self):
        idxs = np.random.permutation(len(self.y))[:self.sample_sz]
        f_idxs = np.random.permutation(self.x.shape[1])[:self.n_features]
        return DecisionTree(self.x[idxs], self.y[idxs], self.n_features, f_idxs,
                    idxs=np.array(range(self.sample_sz)),depth = self.depth, min_leaf=self.min_leaf)
        
    def predict(self, x):
        return np.mean([t.predict(x) for t in self.trees], axis=0)

def std_agg(cnt, s1, s2): return math.sqrt((s2/cnt) - (s1/cnt)**2)

class DecisionTree():
    def __init__(self, x, y, n_features, f_idxs,idxs,depth=10, min_leaf=5):
        self.x, self.y, self.idxs, self.min_leaf, self.f_idxs = x, y, idxs, min_leaf, f_idxs
        self.depth = depth
        self.n_features = n_features
        self.n, self.c = len(idxs), x.shape[1]
        self.val = np.mean(y[idxs])
        self.score = float('inf')
        self.find_varsplit()
        
    def find_varsplit(self):
        for i in self.f_idxs: self.find_better_split(i)
        if self.is_leaf: return
        x = self.split_col
        lhs = np.nonzero(x<=self.split)[0]
        rhs = np.nonzero(x>self.split)[0]
        lf_idxs = np.random.permutation(self.x.shape[1])[:self.n_features]
        rf_idxs = np.random.permutation(self.x.shape[1])[:self.n_features]
        self.lhs = DecisionTree(self.x, self.y, self.n_features, lf_idxs, self.idxs[lhs], depth=self.depth-1, min_leaf=self.min_leaf)
        self.rhs = DecisionTree(self.x, self.y, self.n_features, rf_idxs, self.idxs[rhs], depth=self.depth-1, min_leaf=self.min_leaf)

    def find_better_split(self, var_idx):
        x, y = self.x[self.idxs,var_idx], self.y[self.idxs]
        sort_idx = np.argsort(x)
        sort_y,sort_x = y[sort_idx], x[sort_idx]
        rhs_cnt,rhs_sum,rhs_sum2 = self.n, sort_y.sum(), (sort_y**2).sum()
        lhs_cnt,lhs_sum,lhs_sum2 = 0,0.,0.

        for i in range(0,self.n-self.min_leaf-1):
            xi,yi = sort_x[i],sort_y[i]
            lhs_cnt += 1; rhs_cnt -= 1
            lhs_sum += yi; rhs_sum -= yi
            lhs_sum2 += yi**2; rhs_sum2 -= yi**2
            if i<self.min_leaf or xi==sort_x[i+1]:
                continue

            lhs_std = std_agg(lhs_cnt, lhs_sum, lhs_sum2)
            rhs_std = std_agg(rhs_cnt, rhs_sum, rhs_sum2)
            curr_score = lhs_std*lhs_cnt + rhs_std*rhs_cnt
            if curr_score<self.score: 
                self.var_idx,self.score,self.split = var_idx,curr_score,xi

    @property
    def split_name(self): return self.x.columns[self.var_idx]
    
    @property
    def split_col(self): return self.x[self.idxs,self.var_idx]

    @property
    def is_leaf(self): return self.score == float('inf') or self.depth <= 0 
    

    def predict(self, x):
        return np.array([self.predict_row(xi) for xi in x])

    def predict_row(self, xi):
        if self.is_leaf: return self.val
        t = self.lhs if xi[self.var_idx]<=self.split else self.rhs
        return t.predict_row(xi)

rfcs = RandomForest(X_train, y_train, n_trees=500, n_features='log2', sample_sz=300, depth=20, min_leaf=1)
y_preds = rfcs.predict(X_test)
RF_acc = sum(y_preds == y_test)/len(y_test)
RF_acc = np.round(RF_acc[RF_acc != 0],3)
print("Testing accuracy: %.3f" %max(RF_acc))
