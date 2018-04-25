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

    def split_data(self):
        # tag = 0
        # terminal_no = self.data_from_file.ix[0,"TERMINALNO"]
        # start = 0
        # end = 0
        # while tag<self.data_from_file.shape[0]:
        #     if self.data_from_file.ix[tag,"TERMINALNO"] != terminal_no:
        #         end = tag-1
        #         self.process_personal(start,end)
        #         start = tag
        #         terminal_no = self.data_from_file.ix[tag,"TERMINALNO"]
        #     tag+=1
        #     if tag==self.data_from_file.shape[0]:
        #         end = tag - 1
        #         self.process_personal(start, end)

        for terminal_no,personal_data in self.data_from_file.groupby("TERMINALNO"):
            self.process_personal(terminal_no,personal_data)
        del self.data_from_file
        gc.collect()

    def process_personal(self, terminal_no, personal_data):
        # personal_data = self.data_from_file.ix[start:end,columns]
        # personal_data = personal_data.sort_values(by=["TIME"],ascending=[1])

        personal_data = personal_data.sort_values(by=["TRIP_ID", "TIME"], ascending=[1, 1])
        self.personal_datas[terminal_no] = dict()
        personal_array = np.array(personal_data.ix[:,valid])
        self.personal_datas[terminal_no]["original"] = personal_data
        self.personal_datas[terminal_no]["data"] = personal_array[:,:-1]
        self.personal_datas[terminal_no]["y"] = personal_array[:,-1]
        normed_speed = self.normalize_speed(personal_data)
        # for trip_id,trip_data in personal_data.groupby("TRIP_ID"):
        #     self.personal_datas[terminal_no][trip_id] = trip_data

    def normalize_speed(self,personal_data):
        original_speed = np.array(personal_data.ix[:,"SPEED"])
        normalized_speed_data = self.normalize_min_max(original_speed)
        normalize_speed = pd.DataFrame(data=normalized_speed_data,index=personal_data.index,columns=["SPEED_NORMED"])
        return normalize_speed

    def normalize_min_max(self,data):
        # min max normalize function :

        return data

# if __name__ == "__main__":
#     data = Data(path,columns)
#     time1 = time.time()
#     data.load_data()
#     data.split_data()
#     time2 = time.time()
#     print((time2-time1))
#     print(type(data.data_from_file))
#     # for i in range(data.data_from_file.shape):
