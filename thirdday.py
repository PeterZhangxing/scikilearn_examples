import numpy as np
import pandas as pd

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression,SGDRegressor,Ridge,LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,mean_squared_error
from sklearn.externals import joblib

def my_linear_regression():
    lb = load_boston()
    x_train,x_test,y_train,y_test = train_test_split(lb.data,lb.target,test_size=0.25)

    std_x = StandardScaler()
    std_y = StandardScaler()
    x_train = std_x.fit_transform(x_train)
    x_test = std_x.transform(x_test)
    y_train = std_y.fit_transform(y_train.reshape((-1,1)))
    y_test = std_y.transform(y_test.reshape((-1,1)))

    # 正规方程求解方式预测结果
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    print(lr.coef_)

    joblib.dump(lr,'my_linear_regression.pkl')

    y_lr_predict = std_y.inverse_transform(lr.predict(x_test))
    print("正规方程测试集里面每个房子的预测价格：", y_lr_predict)
    print("正规方程的均方误差：", mean_squared_error(std_y.inverse_transform(y_test), y_lr_predict))

    return None

def my_SGD_regressor():
    lb = load_boston()
    x_train,x_test,y_train,y_test = train_test_split(lb.data,lb.target,test_size=0.25)

    std_x = StandardScaler()
    std_y = StandardScaler()
    x_train = std_x.fit_transform(x_train)
    x_test = std_x.transform(x_test)
    y_train = std_y.fit_transform(y_train.reshape((-1,1)))
    y_test = std_y.transform(y_test.reshape((-1,1)))

    sgd = SGDRegressor()
    sgd.fit(x_train, y_train)
    print(sgd.coef_)

    y_sgd_predict = std_y.inverse_transform(sgd.predict(x_test))
    print("梯度下降测试集里面每个房子的预测价格：", y_sgd_predict)
    print("梯度下降的均方误差：", mean_squared_error(std_y.inverse_transform(y_test), y_sgd_predict))

    return None

def my_ridge():
    lb = load_boston()
    x_train,x_test,y_train,y_test = train_test_split(lb.data,lb.target,test_size=0.25)

    std_x = StandardScaler()
    std_y = StandardScaler()
    x_train = std_x.fit_transform(x_train)
    x_test = std_x.transform(x_test)
    y_train = std_y.fit_transform(y_train.reshape((-1,1)))
    y_test = std_y.transform(y_test.reshape((-1,1)))

    rg = Ridge()
    rg.fit(x_train,y_train)
    print(rg.coef_)

    pre = std_y.inverse_transform(rg.predict(x_test))
    print("测试集里面每个房子的预测价格：", pre)
    print("均方误差：", mean_squared_error(std_y.inverse_transform(y_test), pre))

    return None

def test_joblib():
    lb = load_boston()
    x_train,x_test,y_train,y_test = train_test_split(lb.data,lb.target,test_size=0.25)

    std_x = StandardScaler()
    std_y = StandardScaler()
    x_test = std_x.fit_transform(x_test)
    y_test = std_y.fit_transform(y_test.reshape((-1,1)))

    lr = joblib.load('my_linear_regression.pkl')
    pre = std_y.inverse_transform(lr.predict(x_test))
    print("测试集里面每个房子的预测价格：", pre)
    print("均方误差：", mean_squared_error(std_y.inverse_transform(y_test), pre))

    return None

def my_logistic():
    column = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size',
              'Uniformity of Cell Shape','Marginal Adhesion', 'Single Epithelial Cell Size',
              'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli','Mitoses', 'Class']
    data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
    data = pd.read_csv(data_url,header=None,names=column)
    # print(data.head())
    # print(data.shape[0])
    data.replace(to_replace='?',value=np.nan,inplace=True)
    data.dropna(inplace=True)
    # print(data.shape[0])

    x_train, x_test, y_train, y_test = train_test_split(data[column[1:10]], data[column[10]], test_size=0.25)
    std = StandardScaler()
    x_train = std.fit_transform(x_train)
    x_test = std.transform(x_test)

    lg = LogisticRegression()
    lg.fit(x_train,y_train)
    pre = lg.predict(x_test)

    print(lg.coef_)
    print("准确率：", lg.score(x_test, y_test))
    print('每个类别的精确率和召回率:\n',classification_report(y_test,pre,labels=[2,4],target_names=["良性", "恶性"]))

    return None

if __name__ == '__main__':
    # my_linear_regression()
    # my_SGD_regressor()
    # my_ridge()
    # test_joblib()
    my_logistic()