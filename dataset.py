import numpy as np
import tensorflow as tf
import pandas as pd
import config
import os
import pickle
import pandas as pd
def preprocess(path = "train_data_raw",shuffle = False):
    np.random.seed(1234)
    dataset = []
    labelset = []
    close_price = []
    filtered = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            df = np.array(pd.read_csv(os.path.join(root,file),sep = '\t'))
            data = df[:,3:6][::-1].astype(np.float32)
            for i in range(config.SEQ_LEN,data.shape[0]):
                if (data[i,-1] - data[i-1,-1])/(1e-9+data[i-1,-1]) >=  0.0055 or (data[i,-1] - data[i-1,-1])/(1e-9+data[i,-1]) <=  -0.005:
                    dataset.append(data[i-config.SEQ_LEN:i])
                    label = np.array([np.array([1,0]) if data[j+1,-1] > data[j,-1] else np.array([0,1])
                             for j in range(i-config.SEQ_LEN,i)])
                    labelset.append(np.array(label))
                else:
                    filtered += 1
    labelset = np.array(labelset)
    dataset = np.array(dataset)
    if shuffle:
        ind = np.arange(dataset.shape[0])
        np.random.shuffle(ind)
        dataset = dataset[ind]
        labelset = labelset[ind]
    return dataset, labelset
    #         f = open(os.path.join(root, file_))
    #         for line in f:
    #             count += 1
    #             split_line = line.split('\t')
    #             data.append(np.array([float(i) for i in split_line[3:6]]))
    #             close_price.append(float(split_line[5]))
    #             if data.__len__() >= config.SEQ_LEN:
    #                 dataset.append(np.array(data))
    #                 data.remove(data[0])
    #                 data = []
    # close_price = np.array(close_price)
    # for i in range(close_price.__len__()):
    #     if(i == close_price.__len__() - 1):
    #         label.append(np.array([0,1]))
    #     else:
    #         if(close_price[i+1] > close_price[i]):
    #             label.append(np.array([1,0]))
    #         else:
    #             label.append(np.array([0,1]))
    #     if label.__len__() >= config.SEQ_LEN:
    #         labelset.append(np.array(label))
    #         label.remove(label[0])
            # label = []
    # for i in range(close_price.__len__()):
    #     if(i == close_price.__len__() - 1):
    #         label.append(np.array([1]))
    #     else:
    #         label.append(close_price[i + 1])
    #     if label.__len__() >= config.SEQ_LEN:
    #         labelset.append(np.array(label))
    #         label.remove(label[0])
            # label = []




if __name__ == "__main__":
    # dataset, labelset = preprocess(path = "train_data_raw",shuffle=True)
    # f = open("dataset_train",'wb')
    # pickle.dump(dataset,f)
    # f.close()
    # f = open("labelset_train",'wb')
    # pickle.dump(labelset,f)
    # f.close()
    #
    # dataset, labelset = preprocess(path="test_data_raw", shuffle=False)
    # f = open("dataset_test", 'wb')
    # pickle.dump(dataset, f)
    # f.close()
    # f = open("labelset_test", 'wb')
    # pickle.dump(labelset, f)
    # f.close()

    dataset, labelset = preprocess(path="dev_data_raw", shuffle=False)
    f = open("dataset_dev", 'wb')
    pickle.dump(dataset, f)
    f.close()
    f = open("labelset_dev", 'wb')
    pickle.dump(labelset, f)
    f.close()

