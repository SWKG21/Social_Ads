import pandas as pd
import numpy as np
import multiprocessing as mp
from sklearn.model_selection import train_test_split


input_path = "preliminary_contest_data/"
tmp_path = "tmp_data/"
output_path = "data/"


def sample(frac=0.1):
    train = pd.read_csv(input_path+"train.csv")
    data = train.sample(frac=frac)
    data['label'] = data['label'].replace(-1, 0)
    data.to_csv(tmp_path+"data.csv", index=False)
    train, test = train_test_split(data, test_size=0.3)
    train.to_csv(output_path+"train.csv", index=False)
    test.to_csv(output_path+"test.csv", index=False)


def balanced_sample(frac=1):
    train = pd.read_csv(input_path+"train.csv")
    pos_train = train[train['label']==1]
    neg_train = train[train['label']==-1]
    data = pd.concat([pos_train, neg_train.sample(frac=0.05)]).sample(frac=frac)
    data['label'] = data['label'].replace(-1, 0)
    data.to_csv(tmp_path+"data.csv", index=False)
    train, test = train_test_split(data, test_size=0.3)
    train.to_csv(output_path+"train.csv", index=False)
    test.to_csv(output_path+"test.csv", index=False)


def transform_userFeature():
    ufeas = pd.read_csv(input_path+"userFeature.data", header=None, chunksize=500000)
    for i, ufea in enumerate(ufeas):    # for each chunk
        fea_list = []
        ufea[0] = ufea[0].map(lambda x: x.strip().split('|'))
        for j in range(ufea.shape[0]):  # for each line in the chunk
            fea_dict = {}
            for item in ufea.iloc[j, 0]:    # for each feature in the line
                els = item.split()
                fea_dict[els[0]] = ' '.join(els[1:])
            fea_list.append(fea_dict)
        pd.DataFrame(fea_list).to_csv(tmp_path+"userFeature_"+str(i+1)+".csv", index=False)


def sample_merge_userFeature():
    data = pd.read_csv(tmp_path+"data.csv")
    for i in range(23):
        ufea = pd.read_csv(tmp_path+"userFeature_"+str(i+1)+".csv")
        pd.merge(data, ufea, on='uid', how='inner').drop(columns=['aid', 'label']).to_csv(tmp_path+"userFeature_"+str(i+1)+".csv", index=False)


def merge_userFeature():
    userFea = pd.concat([pd.read_csv(tmp_path+"userFeature_"+str(i+1)+".csv") for i in range(23)])
    userFea.drop_duplicates(inplace=True)
    userFea.to_csv(output_path+"userFeature.csv", index=False)