# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 09:22:34 2022

@author: psdz
"""
import pandas as pd
import numpy as np
import datetime
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt

retail1 = pd.read_excel("./online_retail_II.xlsx", sheet_name = "Year 2009-2010")
retail2 = pd.read_excel("./online_retail_II.xlsx", sheet_name = "Year 2010-2011")
retail = pd.concat([retail1, retail2]).dropna(subset =['Customer ID'])
retail["TotalPrice"] = retail["Quantity"]*retail["Price"]
retail['Customer ID'] = retail['Customer ID'].astype('str')

def cst_seg(data):
    #recency
    today_date = data["InvoiceDate"].max()
    recency = (today_date - data.groupby("Customer ID").agg({"InvoiceDate":"max"}))
    recency['Recency'] = recency['InvoiceDate'].apply(lambda x: x.days)
    recency.drop('InvoiceDate', axis = 1, inplace = True)
    #Frequency
    frequency = data.groupby("Customer ID").agg({"Invoice":"count"})
    frequency.rename(columns={"Invoice": "Frequency"}, inplace = True)
    #Monetary
    monetary = data.groupby("Customer ID").agg({"TotalPrice":"sum"})
    monetary.rename(columns={"TotalPrice": "Monetary"}, inplace = True)
    #join
    return pd.concat([recency, frequency, monetary], axis = 1)

def cluster(rfm_data):
    adj_data = pd.DataFrame()
    for col in rfm_data.columns:
        adj_data[col] = (rfm_data[col]-rfm_data[col].mean())/rfm_data[col].std()
        adj_data[adj_data[col]>3] = 3
        adj_data[adj_data[col]<-3] = -3
    clf = KMeans(n_clusters=5)
    clf.fit(adj_data)
    adj_data['label'] = clf.labels_
    score = []
    center = clf.cluster_centers_
    for i in range(5):
        score.append(-center[i][0]+center[i][1]+center[i][2])
    cent_rfm = {}
    for i in range(5):
        if score[i] == max(score):
            cent_rfm[i] = 3
        elif score[i] == min(score):
            cent_rfm[i] = 1
        else:
            cent_rfm[i] = 2
    adj_data['score'] = adj_data['label'].apply(lambda x: cent_rfm[x])
    return adj_data


ts_cstm = pd.DataFrame(index = list(retail['Customer ID'].unique()))
ts_num = pd.DataFrame(columns = [1,2,3])
for year in (2010,2011):
    for month in range(1,13):
        cut_day = datetime.datetime.strptime(str(year)+str(month), '%Y%m')
        valid_data = retail[retail['InvoiceDate']<cut_day]
        rfm = cst_seg(valid_data)
        clst = cluster(rfm)
        num = clst[['score','label']].groupby('score').count()
        ts_cstm[cut_day] = clst['score']
        ts_num.loc[cut_day] = num['label']




