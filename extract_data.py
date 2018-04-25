# -*- coding:utf8 -*-
import os
import pandas as pd
import gc
import time
import numpy as np

path_train = "/data/dm/train.csv"  # 训练文件
path_test = "/data/dm/test.csv"  # 测试文件

path = "data/dm/train.csv"
columns = ["TERMINALNO", "TIME", "TRIP_ID", "LONGITUDE", "LATITUDE", "DIRECTION", "HEIGHT", "SPEED",
                        "CALLSTATE", "Y"]
valid = ["TIME", "LONGITUDE", "LATITUDE", "DIRECTION", "HEIGHT", "SPEED",
                        "CALLSTATE", "Y"]
path_test_out = "model/"  # 预测结果输出路径为model/xx.csv,有且只能有一个文件并且是CSV格式。


class Data:
    def __init__(self,path,columns):
        self.path = path
        self.columns = columns
        self.personal_datas = {}

    def load_data(self):
        self.data_from_file = pd.read_csv(self.path)
        self.data_from_file.columns = self.columns
        self.data_from_file = self.data_from_file.groupby("TERMINALNO")

    def split_data(self):
        tag = 0
        terminal_no = self.data_from_file.ix[0,"TERMINALNO"]
        start = 0
        end = 0
        while tag<self.data_from_file.shape[0]:
            if self.data_from_file.ix[tag,"TERMINALNO"] != terminal_no:
                end = tag-1
                self.process_personal(start,end)
                start = tag
                terminal_no = self.data_from_file.ix[tag,"TERMINALNO"]
            tag+=1
            if tag==self.data_from_file.shape[0]:
                end = tag - 1
                self.process_personal(start, end)

        del self.data_from_file
        gc.collect()

    def process_personal(self,start,end):
        personal_data = self.data_from_file.ix[start:end,columns]
        # personal_data = personal_data.sort_values(by=["TIME"],ascending=[1])
        nos = []
        # testing part to see if two or more terminal no in a group
        for no in personal_data.ix[:, "TERMINALNO"]:
            if not no in nos:
                nos.append(no)
        if len(nos) == 1:
            self.personal_datas[nos[0]] = dict()
            self.personal_datas[nos[0]]["start"] = start
            self.personal_datas[nos[0]]["end"] = end
            personal_array = np.array(personal_data.ix[:,valid])
            self.personal_datas[nos[0]]["original"] = personal_data
            self.personal_datas[nos[0]]["data"] = personal_array[:,:-1]
            self.personal_datas[nos[0]]["y"] = personal_array[:,-1]


if __name__ == "__main__":
    data = Data(path,columns)
    time1 = time.time()
    data.load_data()
    data.split_data()
    time2 = time.time()
    print((time2-time1))
    print(type(data.data_from_file))
    # for i in range(data.data_from_file.shape):
