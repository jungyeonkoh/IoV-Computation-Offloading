#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
def createLoc_x(t):
    return float(t[0][6:])
def createLoc_y(t):
    return float(t[1][:-1])
file["LOC_X"]=temp_location.map(createLoc_x)
file["LOC_Y"]=temp_location.map(createLoc_y)


# In[2]:


# 2014-02-01부터 2014-02-02까지 데이터 수집
map_data=(file["TIMESTAMP"] >= '2014-02-01') & (file["TIMESTAMP"] <= '2014-02-02')
m=file.loc[map_data]


# In[3]:


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


# In[4]:


m.sort_values(by=["ID", "TIMESTAMP"], inplace=True, ascending=True) 


# In[5]:


m = m.iloc[::4,:]


# In[6]:


import folium

lat = m['LOC_X'].mean()
long = m['LOC_Y'].mean()


# In[7]:


#지도 새로 띄우기
q = folium.Map([lat,long],zoom_start=200)

for i in tqdm(m.index,total=len(m.index)):
    sub_lat =  m.loc[i,'LOC_X']
    sub_long = m.loc[i,'LOC_Y']    
    color = 'green'
        
    #지도에 동그라미로 데이터 찍기    
    folium.CircleMarker([sub_lat,sub_long],color=color,radius = 3).add_to(q)

for i in server_location:
    sub_lat =  i[0]
    sub_long = i[1]
    #
    folium.Marker([sub_lat,sub_long]).add_to(q)
    
#한글이 안나오는 오류로 html로 trouble shooting
q.save('server_and_path_map.html')


# In[ ]:




