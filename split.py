import pandas as pd
import os
import config
def main():
    for root, dirs, files in os.walk(config.DATA_DIR):
        for file_ in files:
            f = open(os.path.join(root, file_))
            df = pd.read_csv(f,sep = '\t')
            test = df[421:485]
            dev = df[485:528]
            train = df[528:925]
            train.to_csv("./train_data_raw/"+file_,sep='\t',index = False)
            dev.to_csv("./dev_data_raw/" + file_, sep='\t', index=False)
            test.to_csv("./test_data_raw/"+file_,sep='\t',index=False)

if __name__ == "__main__":
    main()