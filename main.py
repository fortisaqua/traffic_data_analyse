# -*- coding:utf8 -*-
import os
import csv
import pandas as pd
from model import Model
from extract_data import Data
import argparse

path_train = "/data/dm/train.csv"  # 训练文件
path_test = "/data/dm/test.csv"  # 测试文件

path_test_out = "model/"  # 预测结果输出路径为model/xx.csv,有且只能有一个文件并且是CSV格式。

path = "data/dm/train.csv"
# 加载的数据维度
columns = ["TERMINALNO", "TIME", "TRIP_ID", "LONGITUDE", "LATITUDE", "DIRECTION", "HEIGHT", "SPEED",
           "CALLSTATE", "Y"]
# 有效数据维度
valid = ["TIME", "LONGITUDE", "LATITUDE", "DIRECTION", "HEIGHT", "SPEED",
         "CALLSTATE", "Y"]

def read_csv():
    """
    文件读取模块，头文件见columns.
    :return: 
    """
    # 实例化数据对象
    data = Data(path,columns)
    # 根据选中的维度加载数据，加载columns中的各维度
    data.load_data()
    # 结构化输入数据----{用户编号：{'data':valid中除"Y"以外维度的数据, 'y':valid中"Y"维度的数据}}
    data.split_data(valid)
    return data


def process():
    """
    处理过程，在示例中，使用随机方法生成结果，并将结果文件存储到预测结果路径下。
    :return: 
    """
    import numpy as np
    # 定义参数
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # 模型保存位置
    parser.add_argument('--save_dir', type=str, default='save',
                        help='directory to store checkpointed models')
    # tensorboard 文件保存位置
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='directory to store tensorboard logs')
    # lstm模型中各个门的全连接隐含层大小
    parser.add_argument('--rnn_size', type=int, default=50,
                        help='size of RNN hidden state')
    # lstm cell 的个数
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layers in the RNN')
    # 模型名字
    parser.add_argument('--model', type=str, default='lstm',
                        help='rnn, gru, lstm, or nas')
    # batch大小
    parser.add_argument('--batch_size', type=int, default=30,
                        help='minibatch size')
    # lstm cell中的循环次数
    parser.add_argument('--time_step', type=int, default=30,
                        help='time step num for one loop')
    # 输入数据维度
    parser.add_argument('--input_size', type=int, default=len(valid)-1,
                        help='import single data size')
    # 输出维度
    parser.add_argument('--output_size', type=int, default=1,
                        help='output single data size')
    # 训练批数
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='number of epochs')
    # 保存模型的步数
    parser.add_argument('--save_every', type=int, default=200,
                        help='save frequency')
    # 初始学习率
    parser.add_argument('--learning_rate', type=float, default=0.0006,
                        help='learning rate')
    # 学习率衰减率
    parser.add_argument('--decay_rate', type=float, default=0.97,
                        help='decay rate for rmsprop')
    # lstm cell输出层的droutput保存率，若小于1.0，则为lstm cell添加输入层的droutput层
    parser.add_argument('--output_keep_prob', type=float, default=1.0,
                        help='probability of keeping weights in the hidden layer')
    # lstm cell输入层的droutput保存率，若小于1.0，则为lstm cell添加输出层的droutput层
    parser.add_argument('--input_keep_prob', type=float, default=1.0,
                        help='probability of keeping weights in the input layer')
    parser.add_argument('--init_from', type=str, default=None,
                        help="""continue training from saved model at this path. Path must contain files saved by previous training process:
                                'config.pkl'        : configuration;
                                'chars_vocab.pkl'   : vocabulary definitions;
                                'checkpoint'        : paths to model file(s) (created by tf).
                                                      Note: this file contains absolute paths, be careful when moving files around;
                                'model.ckpt-*'      : file(s) with model definition (created by tf)
                            """)
    args = parser.parse_args()
    args.vocab_size = 1
    # 读取数据并结构化
    data = read_csv()
    # 实例化模型对象
    model = Model(args)
    # 根据args的各参数构建lstm
    model.create_lstm()
    '''
    todo：data类中的加载一个batch的操作（例如：有从batch个用户中各有序的读取一段数据，长度均为time_step）;
          model类中的训练，测试函数（训练函数输出的准确度可使用 gini_confidence.py 中的 gini函数）；
          参考 https://github.com/LouisScorpio/datamining/blob/master/tensorflow-program/rnn/stock_predict/stock_predict_2.py
    '''
    # 测试函数部分，为竞赛模板，未修改
    with open(path_test) as lines:
        with(open(os.path.join(path_test_out, "test.csv"), mode="w")) as outer:
            writer = csv.writer(outer)
            i = 0
            ret_set = set([])
            for line in lines:
                if i == 0:
                    i += 1
                    writer.writerow(["Id", "Pred"])  # 只有两列，一列Id为用户Id，一列Pred为预测结果(请注意大小写)。
                    continue
                item = line.split(",")
                if item[0] in ret_set:
                    continue
                # 此处使用随机值模拟程序预测结果
                writer.writerow([item[0], np.random.rand()]) # 随机值
                
                ret_set.add(item[0])  # 根据赛题要求，ID必须唯一。输出预测值时请注意去重


if __name__ == "__main__":
    print("****************** start **********************")
    # 程序入口
    process()
