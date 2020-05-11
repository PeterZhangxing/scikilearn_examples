import numpy as np
import pandas as pd
import jieba

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA

def dictvec():
    dvector = DictVectorizer(sparse=False)
    raw_data = [
        {'city': '北京','temperature': 100},
        {'city': '上海','temperature':60},
        {'city': '深圳','temperature': 30}
    ]
    data = dvector.fit_transform(raw_data)
    print(dvector.get_feature_names()) # ['city=上海', 'city=北京', 'city=深圳', 'temperature']
    print(data)
    '''
    [[  0.   1.   0. 100.]
    [  1.   0.   0.  60.]
    [  0.   0.   1.  30.]]
    '''
    print(dvector.inverse_transform(data))
    # [{'city=北京': 1.0, 'temperature': 100.0}, {'city=上海': 1.0, 'temperature': 60.0}, {'city=深圳': 1.0, 'temperature': 30.0}]
    return None

def countvec():
    cv = CountVectorizer()
    raw_data = ["人生 苦短， 我 喜欢 python 我 苦短", "人生漫长，不用 python"]
    data = cv.fit_transform(raw_data)
    print(cv.get_feature_names())
    print(data) # sparse vector
    print(data.toarray())
    '''
    /root/PycharmProjects/machine_learning/venv/bin/python /root/PycharmProjects/machine_learning/firstday.py
    ['python', '不用', '人生', '人生漫长', '喜欢', '苦短']
    [[1 0 1 0 1 2]
     [1 1 0 1 0 0]]
    '''
    return None

def cut_chinese(sentence):
    res = ' '.join(list(jieba.cut(sentence)))
    return res

def chinese_vec():
    raw_data = [
        "今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。",
        "我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。",
        "如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。"
    ]
    res_li = []
    for i in raw_data:
        res_li.append(cut_chinese(i))
    # print(res_li)

    cv = CountVectorizer()
    data = cv.fit_transform(res_li)
    print(cv.get_feature_names())
    print(data.toarray())
    return None

def tfidfvec():
    raw_data = [
        "今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。",
        "我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。",
        "如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。"
    ]
    res_li = []
    for i in raw_data:
        res_li.append(cut_chinese(i))

    tfvec = TfidfVectorizer()
    data = tfvec.fit_transform(res_li)
    print(tfvec.get_feature_names())
    print(data.toarray())
    '''
    [[0.         0.         0.21821789 0.         0.         0.
      0.43643578 0.         0.         0.         0.         0.
      0.21821789 0.         0.21821789 0.         0.         0.
      0.         0.21821789 0.21821789 0.         0.43643578 0.
      0.21821789 0.         0.43643578 0.21821789 0.         0.
      0.         0.21821789 0.21821789 0.         0.         0.        ]
     [0.         0.         0.         0.2410822  0.         0.
      0.         0.2410822  0.2410822  0.2410822  0.         0.
      0.         0.         0.         0.         0.         0.2410822
      0.55004769 0.         0.         0.         0.         0.2410822
      0.         0.         0.         0.         0.48216441 0.
      0.         0.         0.         0.         0.2410822  0.2410822 ]
     [0.15698297 0.15698297 0.         0.         0.62793188 0.47094891
      0.         0.         0.         0.         0.15698297 0.15698297
      0.         0.15698297 0.         0.15698297 0.15698297 0.
      0.1193896  0.         0.         0.15698297 0.         0.
      0.         0.15698297 0.         0.         0.         0.31396594
      0.15698297 0.         0.         0.15698297 0.         0.        ]]
    '''
    return None

def mm():
    mm = MinMaxScaler(feature_range=(2, 3))
    raw_data = [[90,2,10,40],
                [60,4,15,45],
                [75,3,13,46]]
    data = mm.fit_transform(raw_data)
    print(data)
    '''
    [[3.         2.         2.         2.        ]
     [2.         3.         3.         2.83333333]
     [2.5        2.5        2.6        3.        ]]
    '''
    return None

def stander():
    stander = StandardScaler()
    raw_data = [[90,2,10,40],
                [60,4,15,45],
                [75,3,13,46]]
    data = stander.fit_transform(raw_data)
    print(data)
    '''
    [[ 1.22474487 -1.22474487 -1.29777137 -1.3970014 ]
     [-1.22474487  1.22474487  1.13554995  0.50800051]
     [ 0.          0.          0.16222142  0.88900089]]
    '''
    return None

def sim():
    im = SimpleImputer(missing_values=np.nan,strategy='mean')
    raw_data = [[1, 2],
                [np.nan, 3],
                [7, 6]]
    data = im.fit_transform(raw_data)
    print(data)
    '''
    [[1. 2.]
     [4. 3.]
     [7. 6.]]
    '''
    return None

def var_sel():
    var = VarianceThreshold(threshold=0)
    raw_data = [[0, 2, 0, 3],
                [0, 1, 4, 3],
                [0, 1, 1, 3]]
    data = var.fit_transform(raw_data)
    print(data)
    '''
    [[2 0]
     [1 4]
     [1 1]]
    '''
    return None

def pca_test():
    pca = PCA(n_components=0.95)
    raw_data = [[2,8,4,5],
                [6,3,0,8],
                [5,4,9,1]]
    data = pca.fit_transform(raw_data)
    print(data)
    '''
    [[ 1.28620952e-15  3.82970843e+00]
     [ 5.74456265e+00 -1.91485422e+00]
     [-5.74456265e+00 -1.91485422e+00]]
    '''
    return None


if __name__ == '__main__':
    # dictvec()
    # countvec()
    # chinese_vec()
    # tfidfvec()
    # mm()
    # stander()
    # sim()
    # var_sel()
    pca_test()