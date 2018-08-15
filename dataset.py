import numpy as np
import tensorflow as tf
import pandas as pd
import config
import os
import pickle
def preprocess(shuffle = False):
    np.random.seed(1234)
    dataset = []
    labelset = []
    data = []
    label = []
    close_price = []
    count = 0
    for root, dirs, files in os.walk(config.DATA_DIR):
        for file_ in files:
            f = open(os.path.join(root, file_))
            for line in f:
                count += 1
                split_line = line.split('\t')
                data.append(np.array([float(i) for i in split_line[3:6]]))
                close_price.append(float(split_line[5]))
                if data.__len__() >= config.SEQ_LEN:
                    dataset.append(np.array(data))
                    data.remove(data[0])
            data = []
    close_price = np.array(close_price)
    for i in range(close_price.__len__()):
        if(i == close_price.__len__() - 1):
            label.append(np.array([0,1]))
        else:
            if(close_price[i+1] > close_price[i]):
                label.append(np.array([1,0]))
            else:
                label.append(np.array([0,1]))
        if i+1 >= config.SEQ_LEN:
            labelset.append(np.array(label))
            label.remove(label[0])
    dataset = np.array(dataset,dtype = np.float)
    labelset = np.array(labelset,dtype=np.float)
    if shuffle:
        ind = np.arange(dataset.shape[0])
        np.random.shuffle(ind)
        dataset = dataset[ind]
        labelset = labelset[ind]
    return dataset,labelset
if __name__ == "__main__":
    dataset, labelset = preprocess(config.DATA_DIR)
    f = open("dataset",'wb')
    pickle.dump(dataset,f)
    f.close()
    f = open("labelset",'wb')
    pickle.dump(labelset,f)
    f.close()

