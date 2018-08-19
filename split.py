import pandas as pd
import os
import config
def main():
    for root, dirs, files in os.walk(config.DATA_DIR):
        for file_ in files:
            f = open(os.path.join(root, file_))
            df = pd.read_csv(f,sep = '\t')
            test = df[:int(df.shape[0]*0.2)]
            train = df[int(df.shape[0]*0.2):]
            train.to_csv("./train_data_raw/"+file_,sep='\t',index = False)
            test.to_csv("./test_data_raw/"+file_,sep='\t',index=False)

if __name__ == "__main__":
    main()