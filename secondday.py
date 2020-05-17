import numpy as np
import pandas as pd

from sklearn.datasets import load_iris,fetch_20newsgroups,load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def test_load_data():
    li = load_iris()
    # print(li.data)
    # print(li.target)
    # print(li.DESCR)
    # x_train,x_test,y_train,y_test = train_test_split(li.data,li.target,test_size=0.25)
    # print("训练集特征值和目标值：", x_train, y_train)
    # print("测试集特征值和目标值：", x_test, y_test)

    # news = fetch_20newsgroups(subset='all')
    # print(news.data)
    # print(news.target)
    # x_train,x_test,y_train,y_test = train_test_split(news.data,news.target,test_size=0.25)
    # print("训练集特征值和目标值：", x_train, y_train)
    # print("测试集特征值和目标值：", x_test, y_test)

    lb = load_boston()
    # print(lb.data)
    # print(lb.target)
    # print(lb.DESCR)
    x_train, x_test, y_train, y_test = train_test_split(lb.data, lb.target, test_size=0.25)
    print("训练集特征值和目标值：", x_train, y_train)
    print("测试集特征值和目标值：", x_test, y_test)
    return None

def knncls():
    data = pd.read_csv("./data/FBlocation/train.csv")
    # (df["name"].str.len() > 1) & (df["age"] > 101)
    # data = data.query("x > 1.0 &  x < 1.25 & y > 2.5 & y < 2.75")
    data = data[(data['x'] > 1.0)&(data['y'] > 2.5)&(data['x'] < 1.25)&(data['y'] < 2.75)]

    time_value = pd.to_datetime(data['time'], unit='s')
    time_value = pd.DatetimeIndex(time_value)
    data['day'] = time_value.day
    data['hour'] = time_value.hour
    data['weekday'] = time_value.weekday

    data = data.drop(['time'], axis=1)
    count = data.groupby(by='place_id')['row_id'].count()
    index = count[count>3].index
    data = data[data['place_id'].isin(index)]
    # data.reindex(index)

    y = data['place_id']
    x = data.drop(['place_id'], axis=1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    std = StandardScaler()
    x_train = std.fit_transform(x_train)
    x_test = std.transform(x_test)

    knn = KNeighborsClassifier()
    # knn.fit(x_train, y_train)
    # y_predict = knn.predict(x_test)
    # print("预测的目标签到位置为：", y_predict)
    # score = knn.score(x_test, y_test)
    # print("预测的准确率:", score)

    params = {'n_neighbors':[3,5,7,10]}
    gc = GridSearchCV(knn,param_grid=params,cv=10)
    gc.fit(x_train,y_train)
    yre = gc.predict(x_test)

    print("预测的目标签到位置为：", yre)
    print("在测试集上准确率：", gc.score(x_test, y_test))
    print("在交叉验证当中最好的结果：", gc.best_score_)
    print("选择最好的模型是：", gc.best_estimator_)
    print("每个超参数每次交叉验证的结果：", gc.cv_results_)
    return None

def navie_bayes():
    news = fetch_20newsgroups(subset='all')
    x_train, x_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.25)
    print(x_train)

    tf = TfidfVectorizer()
    x_train = tf.fit_transform(x_train)
    x_test = tf.transform(x_test)
    print('feature_names:',tf.get_feature_names())
    print(x_train.toarray())

    mlt = MultinomialNB(alpha=1.0)
    mlt.fit(x_train, y_train)
    y_predict = mlt.predict(x_test)
    y_score = mlt.score(x_test, y_test)

    print("预测的文章类别为：", y_predict)
    print("准确率为：", y_score)
    print("每个类别的精确率和召回率：", classification_report(y_test, y_predict, target_names=news.target_names))
    return None

def decision_tree():
    titan = pd.read_csv("http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt")
    x = titan[['pclass', 'age', 'sex']]
    y = titan['survived']
    x['age'].fillna(x['age'].mean(),inplace=True)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    dtvec = DictVectorizer(sparse=False)
    x_train = dtvec.fit_transform(x_train.to_dict(orient="records"))
    x_test = dtvec.transform(x_test.to_dict(orient="records"))
    print(dtvec.get_feature_names())

    dec = DecisionTreeClassifier()
    dec.fit(x_train,y_train)
    pre = dec.predict(x_test)
    score = dec.score(x_test,y_test)
    print("预测的类别为：", pre)
    print("准确率为：", score)
    print("每个类别的精确率和召回率：\n", classification_report(y_test, pre,labels=[0,1],target_names=['dead','survied']))
    return None

def random_forest():
    titan = pd.read_csv("http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt")
    x = titan[['pclass', 'age', 'sex']]
    y = titan['survived']
    x['age'].fillna(x['age'].mean(),inplace=True)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    dtvec = DictVectorizer(sparse=False)
    x_train = dtvec.fit_transform(x_train.to_dict(orient="records"))
    x_test = dtvec.transform(x_test.to_dict(orient="records"))
    print(dtvec.get_feature_names())

    rf = RandomForestClassifier(n_jobs=-1)
    params = {"n_estimators": [120, 200, 300, 500, 800, 1200], "max_depth": [5, 8, 15, 25, 30]}
    gc = GridSearchCV(rf,param_grid=params,cv=10)
    gc.fit(x_train,y_train)

    pre = gc.predict(x_test)
    score = gc.score(x_test,y_test)

    print("预测的类别为：", pre)
    print("准确率为：", score)
    print("查看选择的参数模型：", gc.best_params_)
    print("每个类别的精确率和召回率：\n", classification_report(y_test, pre,target_names=['dead','survied']))
    return None

if __name__ == '__main__':
    # print(test_load_data())
    # decision_tree()
    # random_forest()
    navie_bayes()