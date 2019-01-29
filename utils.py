import pandas as pd
import numpy as np
import multiprocessing as mp
from sklearn.model_selection import train_test_split


input_path = "preliminary_contest_data/"
tmp_path = "tmp_data/"
output_path = "data/"
# cols = ['age', 'gender', 'marriageStatus', 'education', 'consumptionAbility', 'LBS', 
#         'interest1', 'interest2', 'interest3', 'interest4', 'interest5', 'kw1', 'kw2', 'kw3', 
#         'topic1', 'topic2', 'topic3', 'appIdInstall', 'appIdAction', 'ct', 'os', 'carrier', 'house']


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


# def split_userFeature():
#     ufeas = pd.read_csv(input_path+"userFeature.data", header=None, chunksize=3000000)
#     for i, ufea in enumerate(ufeas):
#         ufea[0] = ufea[0].map(lambda x: x.split("|age "))
#         ufea['uid'] = ufea[0].map(lambda x: int(x[0].split()[1]))
#         ufea['fea'] = ufea[0].map(lambda x: "age " + x[1])
#         ufea.drop(columns=[0], inplace=True)
#         ufea.to_csv(tmp_path+"userFeature_"+str(i+1)+".csv", index=False)


# def filter_transform_userFeature(i):
#     data = pd.read_csv(tmp_path+"data.csv")
#     ufea = pd.read_csv(tmp_path+"userFeature_"+str(i+1)+".csv")
#     data_ufea = pd.merge(data, ufea, on='uid', how='inner')
#     ufeas = data_ufea.drop(columns=['aid', 'label'])
#     ufeas.to_csv(tmp_path+"filtered_userFeature_"+str(i+1)+".csv", index=False)
#     del data, ufea, data_ufea, ufeas
#     size = 20000
#     ufeas = pd.read_csv(tmp_path+"filtered_userFeature_"+str(i+1)+".csv", chunksize=size)
#     j = 0
#     for ufea in ufeas:
#         print ("file "+str(i+1)+" part "+str(j+1)+" with shape "+str(ufea.shape))
#         row_num = ufea.shape[0]
#         for col in cols:
#             ufea[col] = [np.NaN for k in range(row_num)]

#         fea = ufea['fea'].map(lambda x: [s.split() for s in x.split('|')])
#         start_row = j * size
#         for row in range(start_row, start_row+row_num):
#             for col in range(len(fea[row])):
#                 if len(fea[row][col]) == 2:
#                     ufea.loc[row, fea[row][col][0]] = fea[row][col][1]
#                 else:
#                     ufea.loc[row, fea[row][col][0]] = ' '.join(fea[row][col][1:])
#         ufea.to_csv(tmp_path+"trans_filtered_userFeature_"+str(i+1)+"_"+str(j+1)+".csv", index=False)
#         j = j + 1
    
#     ufeas = pd.DataFrame(columns=ufea.columns)
#     for k in range(j):
#         tmp = pd.read_csv(tmp_path+"trans_filtered_userFeature_"+str(i+1)+"_"+str(k+1)+".csv")
#         ufeas = pd.concat([ufeas, tmp])
#     ufeas.drop(columns=['fea'], inplace=True)
#     ufeas.to_csv(tmp_path+"trans_filtered_userFeature_"+str(i+1)+".csv", index=False)


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