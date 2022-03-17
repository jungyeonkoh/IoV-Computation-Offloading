import pandas as pd
import numpy as np
from tqdm import tqdm
from haversine import haversine
from multiprocessing import Pool
import multiprocessing as mp
from functools import partial
print("Number of processors: ", mp.cpu_count())


server_location=[(41.9797074301314, 12.517774537227256),
(41.96645724499441, 12.442500521765464),
(41.92362247264185, 12.419533015161926),
(41.859693563285674, 12.424287549712513),
(41.83379583604655, 12.49831550937137),
(41.860946758260795, 12.555531201442555),
(41.94153940428322, 12.553860053429428),
(41.93927042880409, 12.496136876739858),
(41.91832213334783, 12.47306760729969),
(41.887024456510844, 12.472200090859424),
(41.88291790209538, 12.534428295360426),
(41.905086456104385, 12.488293373328746)]

def load_data():
    file = pd.read_csv("./taxi_february.txt", sep=";")
    file = file.rename(columns={ ' TIMESTAMP': 'TIMESTAMP', ' LOCATION': 'LOCATION'})

    # 날짜 추출의 편의를 위해 TIMESTAMP row를 Datetime 항목으로 변경하여 저장
    file["TIMESTAMP"] = pd.to_datetime(file["TIMESTAMP"])
    # 거리 계산의 편의를 위해 LOCATION row를 (float, float) 형식의 튜플로 변경하여 저장
    temp_location = file["LOCATION"].str.split()
    def createLocation(t):
        return (float(t[0][6:]), float(t[1][:-1]))

    file["LOCATION"] = temp_location.map(createLocation)

    # 2014-02-01부터 2014-02-22까지 훈련 데이터
    # 2014-02-23부터 2014-03-02까지 검증 데이터
    train_mask = (file["TIMESTAMP"] >= '2014-02-01') & (file["TIMESTAMP"] <= '2014-02-22')
    test_mask = ~train_mask
    train = file.loc[train_mask]
    test = file.loc[test_mask]

    train.sort_values(by=["ID", "TIMESTAMP"], inplace=True, ascending=True) 
    train=train.reset_index(drop=True)
    test.sort_values(by=["ID", "TIMESTAMP"], inplace=True, ascending=True)
    test=test.reset_index(drop=True)
    
    group_train=train.groupby(train["ID"])
    train_data=group_train.filter(lambda g: len(g)>50000)
    list_train=list(train_data.groupby(train_data["ID"]))
    
    group_test=test.groupby(test["ID"])
    test_data=group_test.filter(lambda g: len(g)>25000)
    list_test=list(test_data.groupby(test_data["ID"]))
    
    return list_train,list_test


def multiprocessing(list_train,num):
    data=list_train[num][1]
    time_inter=[0]
    num=0
    j=1
    prev_timestamp=data.iloc[0]["TIMESTAMP"]
    prev_location=data.iloc[0]["LOCATION"]
    for i, row in tqdm(data.iterrows(),total=data.shape[0]):
        if(haversine(prev_location, row["LOCATION"]) * 1000 == 0 ):
            time_inter.append(0)
        else:
            time_inter.append((row["TIMESTAMP"]-prev_timestamp).seconds)
        prev_timestamp=row["TIMESTAMP"]
        prev_location=row["LOCATION"]
    data["TIME_INTER"]=time_inter[1:]
    data=data.drop(index=data[data["TIME_INTER"]==0].index)
    data=data.drop(columns="TIME_INTER")
    return data

def refine(data):
    re_data=data[0]
    for i in data[1:100]:
        re_data=pd.concat([re_data,i])
    return re_data

# 각 ID의 차량에서 amount개의 데이터만 추출 후 index reset
def preproc(data,amount):    
    data=data.reset_index(drop=True)
    
    j=0
    file=data.iloc[0:amount]
    for i in data.groupby(data["ID"]).size():
        if(j==0):
            j+=i
        else:
            file=pd.concat([file,data[j:j+amount]])
            j+=i
    
    file=file.reset_index(drop=True)
        

    j=0
    for i in tqdm(range(100)):
        file.at[j:amount+j,"ID"]=i+1
        j+=amount
    
    return file

#각 시간별 서버와의 위치, 속도, 추출 후 csv 파일 저장
#[ID, TIMESTAMP, LOCATION, DISTANCE] 형식의 파일로 저장될 예정 DISTANCE는 각 서버에대한 거리를 LIST형식으로 받는다.
#각 시간별 서버와의 위치, 속도, 추출 후 csv 파일 저장
#[ID, TIMESTAMP, LOCATION, DISTANCE] 형식의 파일로 저장될 예정 DISTANCE는 각 서버에대한 거리를 LIST형식으로 받는다.

def get_v(data,server_location,amount,train=True):
    test=data
    prev_id = -1
    
    distance_list=[]
    num=0
    prev_loc=0

    for i, row in tqdm(test.iterrows(),total=test.shape[0]):
        distance_list.append(list(map(lambda x: haversine(x, row["LOCATION"])*1000, server_location)))
        if(row["ID"]!=prev_id):
            prev_id=row["ID"]
            test.at[i,"SPEED"]=0.0
            num=0
        else:
            test.at[i,"SPEED"]=haversine(prev_loc, row["LOCATION"])*1000*3.6/(row["TIMESTAMP"]-prev_time).seconds
        prev_time=row["TIMESTAMP"]
        prev_loc=row["LOCATION"]
        test.at[i,"TIMESTAMP"]=num
        num+=1

    distance_list=np.array(distance_list)
    for i in range(len(server_location)):
        dist_name=("DIST_%s"%(i+1))
        test[dist_name]=distance_list[:,i]
    test=test.drop(index=test[test["SPEED"]==0].index)
    test=test.drop(columns="LOCATION")
    test=test.reset_index(drop=True)
    if train:
        test.to_csv("./train.csv",index = False)
    else:
        test.to_csv("./test.csv",index = False)




if __name__=="__main__":
    print("===============Start preprocessing===============")

    list_train,list_test=load_data()

    pool=Pool(processes= mp.cpu_count())
    func1 = partial(multiprocessing,list_train)
    num_train=list(range(len(list_train)))
    outputs1=pool.map(func1,num_train)
    pool.close()
    pool.join()
    
    pool=Pool(processes= mp.cpu_count())
    func2 = partial(multiprocessing,list_test)
    num_test=list(range(len(list_test)))
    outputs2=pool.map(func2,num_test)
    pool.close()
    pool.join()
    
    train_data=refine(outputs1)
    test_data=refine(outputs2)

    print("===============Get test data===============")
    f=preproc(test_data,min(test_data.groupby(test_data["ID"]).size()))
    get_v(f,server_location,min(test_data.groupby(test_data["ID"]).size()),False)

    print("===============Get train data===============")
    f=preproc(train_data,min(train_data.groupby(train_data["ID"]).size()))
    get_v(f,server_location,min(train_data.groupby(train_data["ID"]).size()),True)
