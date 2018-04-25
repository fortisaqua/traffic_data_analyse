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
def read_csv():
    """
    文件读取模块，头文件见columns.
    :return: 
    """
    # for filename in os.listdir(path_train):
    # tempdata = pd.read_csv(path_train)
    columns = ["TERMINALNO", "TIME", "TRIP_ID", "LONGITUDE", "LATITUDE", "DIRECTION", "HEIGHT", "SPEED",
                        "CALLSTATE", "Y"]
    data = Data(path,columns)
    data.load_data()
    data.split_data()
    return data


def process():
    """
    处理过程，在示例中，使用随机方法生成结果，并将结果文件存储到预测结果路径下。
    :return: 
    """
    import numpy as np
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', type=str, default='data/tinyshakespeare',
                        help='data directory containing input.txt')
    parser.add_argument('--save_dir', type=str, default='save',
                        help='directory to store checkpointed models')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='directory to store tensorboard logs')
    parser.add_argument('--rnn_size', type=int, default=128,
                        help='size of RNN hidden state')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers in the RNN')
    parser.add_argument('--model', type=str, default='lstm',
                        help='rnn, gru, lstm, or nas')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=50,
                        help='RNN sequence length')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='number of epochs')
    parser.add_argument('--save_every', type=int, default=1000,
                        help='save frequency')
    parser.add_argument('--grad_clip', type=float, default=5.,
                        help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.002,
                        help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.97,
                        help='decay rate for rmsprop')
    parser.add_argument('--output_keep_prob', type=float, default=1.0,
                        help='probability of keeping weights in the hidden layer')
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
    data = read_csv()
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
