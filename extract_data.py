# -*- coding:utf8 -*-
import os
import csv
import pandas as pd
import gc
import time

path_train = "/data/dm/train.csv"  # 训练文件
path_test = "/data/dm/test.csv"  # 测试文件

path = "./PINGAN-2018-train_demo.csv"
columns = ["TERMINALNO", "TIME", "TRIP_ID", "LONGITUDE", "LATITUDE", "DIRECTION", "HEIGHT", "SPEED",
                        "CALLSTATE", "Y"]
valid = ["TIME", "LONGITUDE", "LATITUDE", "DIRECTION", "HEIGHT", "SPEED",
                        "CALLSTATE", "Y"]
path_test_out = "model/"  # 预测结果输出路径为model/xx.csv,有且只能有一个文件并且是CSV格式。


class Data:
    def __init__(self,path,columns):
        self.path = path
        self.columns = columns
        self.personal_data = {}

    def load_data(self):
        self.data_from_file = pd.read_csv(self.path)
        self.data_from_file.columns = self.columns
        self.data_from_file.sort_values(by=["TERMINALNO","TRIP_ID"],ascending=[1,1])

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
        nos = []
        # testing part to see if two or more terminal no in a group
        for no in personal_data.ix[:,"TERMINALNO"]:
            if not no in nos:
                nos.append(no)
        if len(nos) == 1:
            self.personal_data[nos[0]] = dict()
            self.personal_data[nos[0]]["start"] = start
            self.personal_data[nos[0]]["end"] = end
            print("processing personal data from ", start, " to ", end, " : ", nos)
            # print(personal_data)
            tag = start
            trip_id = personal_data.ix[start, "TRIP_ID"]
            start_per = start
            end_per = start
            while tag < end:
                if personal_data.ix[tag, "TRIP_ID"] != trip_id:
                    end_per = tag - 1
                    self.process_single_trip(personal_data,start_per, end_per)
                    start_per = tag
                    trip_id = personal_data.ix[tag, "TRIP_ID"]
                tag += 1
                if tag == end:
                    end_per = tag - 1
                    self.process_single_trip(personal_data,start_per, end_per)

    def process_single_trip(self,personal_data,start,end):
        trip_data = personal_data.ix[start:end,columns]
        ids = []
        for id in trip_data.ix[:,"TRIP_ID"]:
            if not id in ids:
                ids.append(id)
        print("processing single trip data from ", start, " to ", end, " : ", ids)


if __name__ == "__main__":
    data = Data(path,columns)
    time1 = time.time()
    data.load_data()
    data.split_data()
    time2 = time.time()
    print((time2-time1))
    # print(data.data_from_file.ix[:30 ,[]])
    # for i in range(data.data_from_file.shape):
