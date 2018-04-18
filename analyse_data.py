import numpy
import csv
import cPickle as pickle
import pandas as pd

train_dir = "./PINGAN-2018-train_demo.csv"
data_dir = "./train.pkl"
cell_list = []
data = dict()
counter = 0

with open(train_dir,'rb') as f:
    reader = csv.DictReader(f)
    for row in reader:
        cell_list.append(row)
        terminal_no = row["TERMINALNO"]
        trip_id = row["TRIP_ID"]
        if not data.has_key(terminal_no):
            data[terminal_no]=dict()
        if not data[terminal_no].has_key(trip_id):
            data[terminal_no][trip_id] = list()
        del row["TERMINALNO"]
        del row["TRIP_ID"]
        data[terminal_no][trip_id].append(row)
        counter += 1
pickle_writer = open(data_dir, 'wb')
pickle.dump(data, pickle_writer)
pickle_writer.close()