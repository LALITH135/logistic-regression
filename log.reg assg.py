


import pandas as pd

df = pd.read_csv('C:\\python notes\\ASSIGNMENTS\\log reg\\bank-full.csv',sep=';')
df.shape
list(df)
df.head()
df.info()

 # label encoder
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df['job'] = LE.fit_transform(df['job'])
df['job'].value_counts()

from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df['marital'] = LE.fit_transform(df['marital'])
df['marital'].value_counts()

from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df['education'] = LE.fit_transform(df['education'])
df['education'].value_counts()

from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df['default'] = LE.fit_transform(df['default'])
df['default'].value_counts()

from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df['housing'] = LE.fit_transform(df['housing'])
df['housing'].value_counts()

from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df['loan'] = LE.fit_transform(df['loan'])
df['loan'].value_counts()

from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df['contact'] = LE.fit_transform(df['contact'])
df['contact'].value_counts()

from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df['month'] = LE.fit_transform(df['month'])
df['month'].value_counts()

from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df['poutcome'] = LE.fit_transform(df['poutcome'])
df['poutcome'].value_counts()

from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df['y'] = LE.fit_transform(df['y'])
df['y'].value_counts()


# split the variables in X and Y
Y = df['y']
X = df.iloc[:,0:15]

# standardization on X data
from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
X_scale = SS.fit_transform(X)

# model fitting  --> logistic regression
# X_scale and Y
from sklearn.linear_model import LogisticRegression
LogR = LogisticRegression()
LogR.fit(X_scale,Y)
Y_pred = LogR.predict(X_scale)
Y_pred

#confusion matrix

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y,Y_pred)
print(cm)
# [[39151   771]
# [ 4178  1111]]


TN = cm[0,0]  # TN
TP = cm[1,1]  # TP
FP = cm[0,1]  # FP
FN = cm[1,0]  # FN
TNR = TN / (TN + FP)
print("specificity score: ", TNR.round(3))
#specificity score:  0.981

from sklearn.metrics import accuracy_score, recall_score
ac = accuracy_score(Y,Y_pred)
print("Accuracy score: ", ac.round(3))
#Accuracy score:  0.891

rc = recall_score(Y,Y_pred)
print("Sensitivity score: ", rc.round(3))
#Sensitivity score:  0.21

from sklearn.metrics import precision_score, f1_score
ps = precision_score(Y,Y_pred)
print("precision_score: ", ps.round(3))
#precision_score:  0.59

f1s = f1_score(Y,Y_pred)
print("f1_score: ", f1s.round(3))
#f1_score:  0.31

#scatter plot

import seaborn as sns

P=df.iloc[:,11]
Q=df.iloc[:,16]

sns.regplot(x=P, y=Q, data=df, logistic=True, ci=None)

#=========================================================================================









