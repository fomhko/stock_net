import numpy as np
import tensorflow as tf
import pandas as pd
import config
def preprocess(datafile,shuffle = False):
    f = open(datafile)
    dataset = []
    labelset = []
    data = []
    label = []
    close_price = []
    count = 0
    for line in f:
        count += 1
        split_line = line.split('\t')
        data.append(np.array([float(i) for i in split_line[3:6]]))
        close_price.append(float(split_line[5]))
        if count >= config.SEQ_LEN:
            dataset.append(np.array(data))
            data.remove(data[0])
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
    preprocess(config.DATA_DIR+"AAPL.txt")

