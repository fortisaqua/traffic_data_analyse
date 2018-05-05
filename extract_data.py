# -*- coding:utf8 -*-
import os
import pandas as pd
import gc
import time
import numpy as np

class Data:
    def __init__(self, path, columns):
        self.path = path
        self.columns = columns
        self.personal_datas = {}

    def load_data(self):
        self.data_from_file = pd.read_csv(self.path)
        self.data_from_file.columns = self.columns

    # 根据用户编号分裂数据并结构化
    def split_data(self,valid):
        for terminal_no,personal_data in self.data_from_file.groupby("TERMINALNO"):
            self.process_personal(terminal_no, personal_data, valid)
        del self.data_from_file
        gc.collect()

    # 结构化个人数据，根据行程编号和时间排序
    def process_personal(self, terminal_no, personal_data, valid):

        personal_data = personal_data.sort_values(by=["TRIP_ID", "TIME"], ascending=[1, 1])
        self.personal_datas[terminal_no] = dict()
        personal_array = np.array(personal_data.ix[:,valid])
        # self.personal_datas[terminal_no]["original"] = personal_data
        self.personal_datas[terminal_no]["data"] = personal_array[:,:-1]
        self.personal_datas[terminal_no]["y"] = personal_array[:,-1]
        # normed_speed = self.normalize_speed(personal_data)
        # for trip_id,trip_data in personal_data.groupby("TRIP_ID"):
        #     self.personal_datas[terminal_no][trip_id] = trip_data

    # 根据用户个人数据归一化速度数据，未使用
    def normalize_speed(self,personal_data):
        original_speed = np.array(personal_data.ix[:,"SPEED"])
        normalized_speed_data = self.normalize_min_max(original_speed)
        normalize_speed = pd.DataFrame(data=normalized_speed_data,index=personal_data.index,columns=["SPEED_NORMED"])
        return normalize_speed

    # 归一化速度数据函数，未实现
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
