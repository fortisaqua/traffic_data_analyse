import cPickle as pickle

data_dir = './train.pkl'
pickle_reader = open(data_dir,'rb')
data = pickle.load(pickle_reader)

# method of extract message
new_counter = 0

for no,group in data.items():
    if not group.has_key("compensate"):
        data[no]["compensate"] = list()
    for id,elem in group.items():
        if not id == "compensate":
            new_counter+=len(elem)
            for component in elem:
                compensate_num = component["Y"]
                if not compensate_num in data[no]["compensate"]:
                    data[no]["compensate"].append(compensate_num)

# analyse compensation rate
max_compensation_len = 0
for no,group in data.items():
    if len(group["compensate"])>max_compensation_len:
        max_compensation_len = len(group["compensate"])

print("done")