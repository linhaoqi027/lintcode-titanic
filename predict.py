import  pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

train_data=pd.read_csv('train.csv')
test_data=pd.read_csv('test.csv')
#.loc[train_data["Sex"]=="male","Sex"]=0
#train_data.loc[train_data["Sex"]=="female","Sex"]=1

train_data["Age"]=train_data["Age"].fillna(train_data["Age"].mean())//1 #//1对年龄取整数
test_data["Age"]=test_data["Age"].fillna(train_data["Age"].mean())//1

train_data["Embarked"]=train_data["Embarked"].fillna('S')
test_data["Embarked"]=train_data["Embarked"].fillna('S')
sub1 = test_data.PassengerId
train_data.loc[train_data["Fare"]==0,"Fare"]=np.nan
test_data.loc[test_data["Fare"]==0,"Fare"]=np.nan

for i in range(train_data["Pclass"].max()):
    train_data.loc[train_data.Pclass == i & train_data.Fare.isnull(),"Fare"]=train_data.loc[train_data.Pclass==i,["Fare"]].mean()
    test_data.loc[test_data.Pclass == i & test_data.Fare.isnull(), "Fare"] = test_data.loc[test_data.Pclass == i, ["Fare"]].mean()

train_data.drop(['PassengerId','Name','Cabin','Ticket'],axis=1,inplace=True)
test_data.drop(['PassengerId','Name','Cabin','Ticket'],axis=1,inplace=True)

y = train_data.Survived
# using one hot for types of Object
X= train_data.drop(['Survived'],axis=1)
X = pd.get_dummies(X)

test_X = test_data

test_X = pd.get_dummies(test_X)
test_X.to_csv('x1.csv')
X,test_X = X.align(test_X,join='left',axis=1)
test_X.to_csv('x2.csv')
test_X = test_X.values
train_X, valid_X, train_y, valid_y = train_test_split(X.values, y.values, test_size=0.3)

clf = XGBClassifier(max_depth=5,
                        min_child_weight=1,
                        learning_rate=0.1,
                        n_estimators=500,
                        silent=True,
                        objective='binary:logistic',
                        gamma=0,
                        max_delta_step=0,
                        subsample=1,
                        colsample_bytree=1,
                        colsample_bylevel=1,
                        reg_alpha=0,
                        reg_lambda=0,
                        scale_pos_weight=1,
                        seed=1,
                        missing=None)

clf.fit(train_X, train_y, eval_metric='auc', verbose=False,
            eval_set=[(valid_X, valid_y)], early_stopping_rounds=10)

y_pre = clf.predict(valid_X)
y_pro = clf.predict_proba(valid_X)[:, 1]#预测属于某一类别的概率
print("AUC Score : %f" % metrics.roc_auc_score(valid_y, y_pro))
print("Accuracy : %.4g" % metrics.accuracy_score(valid_y, y_pre))

predictions = clf.predict(test_X)
pre = np.round(predictions).astype(int)
subm = pd.DataFrame({'PassengerId':sub1,'Survived':pre})
subm.to_csv('mytitan.csv', index = False)

