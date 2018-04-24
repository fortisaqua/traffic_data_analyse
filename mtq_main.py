#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import csv
import pandas as pd
import numpy as np
from sklearn.tree import tree
from sklearn.externals import joblib
from collections import defaultdict
from sklearn import linear_model
from sklearn import svm
from sklearn import ensemble
from sklearn import neural_network
path_train = "/data/dm/train.csv"  # 训练文件
path_test = "/data/dm/test.csv"  # 测试文件

path_test_out = "model/"  # 预测结果输出路径为model/xx.csv,有且只能有一个文件并且是CSV格式。

def read_csv(path):
    """
    文件读取模块，头文件见columns.
    :return:
    """
    # for filename in os.listdir(path_train):
    tempdata = pd.read_csv(path, dtype=float)
    tempdata.columns = ["TERMINALNO", "TIME", "TRIP_ID", "LONGITUDE", "LATITUDE", "DIRECTION", "HEIGHT", "SPEED",
                        "CALLSTATE", "Y"]
    tempdata = np.array(tempdata)
    return tempdata[1:, :]


def regression(data, clf_label):
    res_data = []
    for i in range(data.shape[0]):
        if clf_label[i] != 0:
            res_data.append(data[i, ...].tolist())
    res_data = np.array(res_data)
    return res_data


def process():
    """
    处理过程，在示例中，使用随机方法生成结果，并将结果文件存储到预测结果路径下。
    :return:
    """
    path = "./PINGAN-2018-train_demo.csv"
    data = read_csv(path)

    print("aaaaaaaaaaaaaaaaaaaaa0")
    #分类(决策树)
    classify_label = np.zeros(data[:, -1].shape)
    print("aaaaaaaaaaaaaaaaaaaaa1.1")
    classify_label[data[:, -1] != 0.0] = 1

    clf_classify = svm.SVC()
    clf_classify.fit(data[:, :-1], classify_label)
    print("aaaaaaaaaaaaaaaaaaaaa1")
    # regresion_data = regression(data[:, :-1], classify_label)
    # regresion_label = regression(data[:, -1], classify_label)
    print("aaaaaaaaaaaaaaaaaaaaa2")


    #回归（随机森林）
    clf_regressor = ensemble.RandomForestRegressor()
    clf_regressor.fit(data[:, :-1], data[:, -1])

    del classify_label
    # del regresion_data
    # del regresion_label
    print("aaaaaaaaaaaaaaaaaaaaa3")

    test_data = pd.read_csv(path_test, header=0, dtype=float)
    test_data = np.array(test_data)
    row, col = test_data.shape

    test_result_classify = clf_classify.predict(test_data)

    # test_regress_data = regression(test_data, test_result_classify)
    print("aaaaaaaaaaaaaaaaaaaaa4")
    test_result_regressor = clf_regressor.predict(test_data)
    ind = 0
    for i in range(test_result_classify.shape[0]):
        if test_result_classify[i] != 0:
            test_result_classify[i] = test_result_regressor[i]
            # ind += 1
    print("aaaaaaaaaaaaaaaaaaaa5")

    #test_result = np.zeros(test_result_classify.shape)
    peifulv = defaultdict(float)
    # index = 1
    # peifulv[(index)] = test_result[0]
    # for i in range(1,row):
    #     #后面的索引不等于前面的索引
    #     if test_data[i,0]!=test_data[i-1,0]:
    #         index+=1#索引加一
    #         peifulv[(index)] = test_result[i]
    #     else:
    #     # 后面的索引等于前面的索引
    #         if test_result[i]<0:
    #             peifulv[(index)]=0
    #         elif test_result[i]>test_result[i-1]:
    #             peifulv[(index)] = test_result[i]
    #         else:
    #             continue
    for i in range(row):
        if peifulv[(test_data[i, 0])] < test_result_classify[i]:
            peifulv[(test_data[i, 0])] = test_result_classify[i]

    print(len(peifulv))
    # for i in range(1,index):
    #     peifulv[(i)] = (peifulv[(i)]+1)/2
    # print(index)

    with(open(os.path.join(path_test_out, "test.csv"), mode="w")) as outer:
        writer = csv.writer(outer)
        writer.writerow(["Id", "Pred"])  # 只有两列，一列Id为用户Id，一列Pred为预测结果(请注意大小写)。
        for line in peifulv:
            writer.writerow([line, peifulv[(line)]])


if __name__ == "__main__":
    print("****************** start **********************")
    # 程序入口
    process()
