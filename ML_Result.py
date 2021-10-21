from xgboost import XGBClassifier
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score,roc_auc_score,make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from xgboost import plot_tree
from tensorflow.keras import layers
from tensorflow.keras import activations
from sklearn.neural_network import MLPClassifier
import math


data=pd.read_csv('cheating-noncheating-combined.csv') #combined csv

data = data.drop(['label 1'], axis = 1)
data = data.drop(['label 2'], axis = 1)
data = data.drop(['time'], axis = 1)

print(data.head())
print(data.shape)

data.isnull().sum()

#minmax scalling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

y_new= data['label'].copy()  
x_new=pd.DataFrame(data.drop(['label'],axis=1))
x_new_MLP=pd.DataFrame(data.drop(['label'],axis=1))
#checking correlation
data_corr = data.corr()
import seaborn as sns
sns.heatmap(data_corr, cmap = 'copper')
#removing one of the two features having correaltion greater than 0.9
columns = np.full((data_corr.shape[0],), True, dtype=bool)
for i in range(data_corr.shape[0]):
    for j in range(i+1, data_corr.shape[0]):
        if data_corr.iloc[i,j] >= 0.9:
            if columns[j]:
                columns[j] = False
selected_columns = data.columns[columns]
data = data[selected_columns]
data.shape

#checking correlation again
data_corr = data.corr()
import seaborn as sns
sns.heatmap(data_corr, cmap = 'copper')

#hybrid Model

#XGBoost

X_train,X_test,y_train,y_test= train_test_split(x_new,y_new,test_size=0.2,random_state=42)
hybridXGB=XGBClassifier(max_depth=10,subsample=1,n_estimators=250,learning_rate=0.05)
start = time.time()
hybridXGB.fit(X_train,y_train)
stop = time.time()
print(stop - start)
hybridPredit=hybridXGB.predict(x_new)
x_new_MLP['predict']=hybridPredit

#MLP

X_trainMLP,X_testMLP,y_trainMLP,y_testMLP= train_test_split(x_new_MLP,y_new,test_size=0.2,random_state=1)
hybridMLP=MLPClassifier(solver='lbfgs', alpha=1e-5,
                      hidden_layer_sizes=(45,25), random_state=15,max_iter=5000000,max_fun=1500000,activation='relu')


start = time.time()
hybridMLP.fit(X_trainMLP,y_trainMLP)
stop = time.time()
print(stop - start)
hybridPreditFinal=hybridMLP.predict(X_testMLP)
y_train_predict=hybridMLP.predict(X_trainMLP)

#Result

print("Train accuracy",accuracy_score(y_trainMLP,y_train_predict)*100)
hybridAccuracy=accuracy_score(y_testMLP,hybridPreditFinal)*100
print("Test Accuracy",hybridAccuracy)
pd.crosstab(y_testMLP,hybridPreditFinal)
plot_confusion_matrix(hybridMLP,X_testMLP,y_testMLP,display_labels=["Did not cheat","Cheat"],cmap=plt.cm.pink,normalize='pred')