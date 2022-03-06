#!/usr/bin/env python
# coding: utf-8

# In[1]:


# %pip install haversine # 필요 패키지


# In[2]:


import pandas as pd
import numpy as np
from tqdm import tqdm
from haversine import haversine

file = pd.read_csv("./taxi_february.txt", sep=";")
file = file.rename(columns={ ' TIMESTAMP': 'TIMESTAMP', ' LOCATION': 'LOCATION'})

# 날짜 추출의 편의를 위해 TIMESTAMP row를 Datetime 항목으로 변경하여 저장
file["TIMESTAMP"] = pd.to_datetime(file["TIMESTAMP"])
# 거리 계산의 편의를 위해 LOCATION row를 (float, float) 형식의 튜플로 변경하여 저장
temp_location = file["LOCATION"].str.split()
def createLocation(t):
    return (float(t[0][6:]), float(t[1][:-1]))

file["LOCATION"] = temp_location.map(createLocation)


# In[3]:


# 2014-02-01부터 2014-02-15까지 훈련 데이터
# 2014-02-16부터 2014-03-02까지 검증 데이터
train_mask = (file["TIMESTAMP"] >= '2014-02-01') & (file["TIMESTAMP"] <= '2014-02-15')
test_mask = ~train_mask
train = file.loc[train_mask]
test = file.loc[test_mask]


# In[4]:


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


# In[5]:


train.sort_values(by=["ID", "TIMESTAMP"], inplace=True, ascending=True) # 총 8,300,961개의 columns


# In[6]:


test.sort_values(by=["ID", "TIMESTAMP"], inplace=True, ascending=True) # 총 8,300,961개의 columns


# In[7]:


#20000개 이상의 데이터를 갖고 있지 않는 데이터 거르기.
group_train=train.groupby(train["ID"])
train_data=group_train.filter(lambda g: len(g)>20000)

group_test=test.groupby(train["ID"])
test_data=group_train.filter(lambda g: len(g)>20000)


num_train=set(list(train_data["ID"]))
num_test=set(list(test_data["ID"]))

num_ID=num_train and num_test  # train데이터와 test데이터에서 둘다 20000개를 넘게 갖고있는 데이터의 ID


# In[8]:


#Train 데이터에서 20000개 이상의 데이터중 200개의 차량 선택

data_train=train_data
num_train=num_train - set(list(num_ID)[:200])
for i in list(num_train):
    data_train=data_train.drop(index=data_train[data_train["ID"]==i].index)


# In[9]:


#Test 데이터에서 20000개 이상의 데이터중 200개의 차량 선택


data_test=test_data
num_test=num_test - set(list(num_ID)[:200])
for i in list(num_test):
    data_test=data_test.drop(index=data_test[data_test["ID"]==i].index)


# In[ ]:





# In[ ]:





# In[10]:


# 각 ID의 차량에서 20001개의 데이터만 추출 후 index reset
def preproc(data):    
    data=data.reset_index(drop=True)
    
    j=0
    file=data.iloc[0:20001]
    for i in data.groupby(data["ID"]).size():
        if(j==0):
            j+=i
        else:
            file=pd.concat([file,data[j:j+20001]])
            j+=i
    
    file=file.reset_index(drop=True)
        

    j=0
    k=1
    for i in tqdm(range(200)):
        file.at[j:20001+j,"ID"]=k
        k+=1
        j+=20001
    
    return file


# In[11]:


#각 시간별 서버와의 위치, 속도, 추출 후 csv 파일 저장
#[ID, TIMESTAMP, LOCATION, DISTANCE] 형식의 파일로 저장될 예정 DISTANCE는 각 서버에대한 거리를 LIST형식으로 받는다.

def get_v(data,server_location,train=True):
    test=data
    prev_id = -1

    idx_1 = -1
    idx_2 = -1
    distance_list=[]
    num=0
    for i, row in tqdm(test.iterrows(),total=test.shape[0]):
        distance_list.append(list(map(lambda x: haversine(x, row["LOCATION"]), server_location)))

        if row["ID"] == prev_id:
            time_diff = row["TIMESTAMP"] - prev_timestamp
            time_diff = time_diff.seconds
            distance = haversine(prev_location, row["LOCATION"]) * 1000 # 미터 단위로
            
            try:
                speed = distance * 3.6 / time_diff # km/h 단위로
                test.at[i, "SPEED"] = round(speed, 3)

            except ZeroDivisionError:
                # 이전 두 개의 데이터의 평균값으로 대체
                if idx_2 == -1:
                    test.at[i, "SPEED"] = 0.0
                elif test.at[idx_2, "ID"] == prev_id:
                    speed = (test.at[idx_1, "SPEED"] + test.at[idx_2, "SPEED"]) / 2
                    test.at[i, "SPEED"] = round(speed, 3)
                else:
                    test.at[i, "SPEED"] = 0.0

            if (test.at[i, "SPEED"] == 0.0):
                # 이전 두 개의 데이터의 평균값으로 대체
                if idx_2 == -1:
                    test.at[i, "SPEED"] = 0.0
                elif test.at[idx_2, "ID"] == prev_id:
                    speed = (test.at[idx_1, "SPEED"] + test.at[idx_2, "SPEED"]) / 2
                    test.at[i, "SPEED"] = round(speed, 3)
                else:
                    test.at[i, "SPEED"] = 0.0

        # 각 차량별 초기데이터는 속도를 구할수 없으므로 NAN으로 처리
        else:
            prev_id = row["ID"]
            test.at[i, "SPEED"] = np.nan

        idx_2 = idx_1
        idx_1 = i
        prev_timestamp = row["TIMESTAMP"]
        prev_location = row["LOCATION"]
        test.at[i,"TIMESTAMP"]=num
        num+=1
        if (num==20001):
            num=0

    test['DISTANCE']=distance_list
    test = test.drop(index=test[test["TIMESTAMP"]==0].index)


    if train:
        test.to_csv("./train.csv",index = False)
    else:
        test.to_csv("./test.csv",index = False)


# In[12]:


f=preproc(data_test)
get_v(f,server_location,False)
f=preproc(data_train)
get_v(f,server_location,True)


# In[ ]:




