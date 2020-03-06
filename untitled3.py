# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 21:45:57 2020

# -*- encoding=utf-8 -*-
import xlsxwriter
import pandas as pd
from pandas import DataFrame
from itertools import permutations
from text_anti_brush_function import *
df=pd.read_excel('Check_Data.xlsx')
df1=df["辅助核查列"]
df2=df["汇总ID"]

content_list=df1

source_id_list=df2
data1=pd.DataFrame({'content':content_list,'source_id':source_id_list})


#print(data1)
test_data=dict(data1['content'])
print('排列有:')
k1=[]
k2=[]
k3=[]
k4=[]
k5=[]

for i,j in permutations(test_data, 2):
    similar=lcs_similarity(str(data1.iloc[i,0]),str(data1.iloc[j,0]))
    #print(i,j,data1.iloc[i,0],data1.iloc[j,0],similar)
    df3=data1.iloc[i,0]
    df4=data1.iloc[j,0]
    k1.append(i)
    k3.append(j)
    k4.append(df3)
    k5.append(df4)
    k2.append(similar)
    
df5=[x + 1 for x in k1]
df6=[x + 1 for x in k3]
data3=pd.DataFrame({'k1':df5,'k4':k4,'k3':df6,'k5':k5,'k2':k2})

data3_103 = data3[data3['k2'] >= 0.75]
print(data3_103)

data3_103.to_excel(r'Match.xlsx')
