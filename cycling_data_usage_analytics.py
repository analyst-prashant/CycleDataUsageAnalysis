#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing necessary libraries
import numpy as np
import pandas as pd
import requests, os, re ,csv, json, glob
import urllib.request
import urllib.parse
import networkx as nx

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

import tkinter as tk
from tkinter import filedialog

import datetime

import time
import warnings
warnings.filterwarnings("ignore")


# In[2]:


path = "C:/Users/prash/Documents/cyclingdata/usage-data.csv"


# In[3]:


df = pd.read_csv(path)


# The TfL bike usage data is hosted as a number of CSV files on the website. 

# In[4]:


df['File'] = df['File'].str[1:]
# trimming data from 2019 to 2021 (some files from different year are also included)
# given in the problem statement - assuming data start with the file 143JourneyDataExtract02Jan2019-08Jan2019.csv
df = df[67:243]


# In[5]:


df.head()


# In[6]:


df.tail()


# In[7]:


#Create function to pull the csv based on name
def parse_url(l): 
    url = "http://cycling.data.tfl.gov.uk/usage-stats/{}".format(urllib.parse.quote(l[:]))
    print(url)
    urllib.request.urlretrieve(url, l[:])
    print('Done')


# In[52]:


# %% time
[parse_url(l) for  l in df['File']]


# In[8]:


# Use Tinker to select target folder
# Select the folder where the files are downloaded

root = tk.Tk()
root.withdraw()
dirname = filedialog.askdirectory(initialdir="C:/Users/prash/Documents/cyclingdata",title='Please select a directory')
if len(dirname ) > 0:
    print("The directory is %s" % dirname)


# In[9]:


def pullbatch(searchstring):
    all_files = glob.glob(os.path.join(dirname , searchstring))
    l = []
    
    for i, filename in enumerate(all_files):
        # print("Reading ", i , " csv file" )
        df = pd.read_csv(filename, index_col=None, header=0)
        df['filename'] = filename
        df['filename'] = df['filename'].str.replace(dirname,'')
        df['Duration(Min)'] = df['Duration'] / 60.00
        l.append(df)
        # print("Done")
    df_new = pd.concat(l, axis=0, ignore_index=True)
    print("Done")
    return df_new
    


# In[10]:


get_ipython().run_cell_magic('time', '', 'df2019 = pullbatch("*2019.csv")\n')


# In[11]:


df2019.info()


# In[12]:


get_ipython().run_cell_magic('time', '', 'df2020 = pullbatch("*2020.csv")\n')


# In[13]:


df2020.info()


# In[14]:


get_ipython().run_cell_magic('time', '', 'df2021 = pullbatch("*2021.csv")\n')


# In[15]:


df2021.info()


# In[16]:


# Combine all bike-share data for the tfl, January 2020 up until December 2021
combine_df = [df2019, df2020, df2021]  # List of dataframes
final_df = pd.concat(combine_df)


# In[17]:


# Copying data from final_df for later use
df_new = final_df


# In[18]:


print(final_df.shape)

final_df.dropna(axis=0, subset=["StartStation Id", "EndStation Id", "Start Date", "End Date"], inplace=True)

print(final_df.shape)


# In[19]:


# Cleaning data - drop additional useless columns
final_df = final_df[final_df["StartStation Id"] != final_df["EndStation Id"]]

final_df = final_df.loc[:,('Start Date', 'StartStation Id', 'End Date', 'EndStation Id', 'Duration')]
                           
print(final_df.shape)


# In[20]:


## Dropping duplicates
final_df.drop_duplicates(inplace=True)
print(final_df.shape)


# In[21]:


final_df.head()


# In[22]:


final_df.info()


# In[23]:


final_df['StartStation Id'].value_counts()


# In[24]:


final_df['EndStation Id'].value_counts()


# In[ ]:





# In[25]:


# Changing object datatype to datetime data 
# final_df['End Date']= pd.to_datetime(final_df['End Date'])
final_df['Start Date']= pd.to_datetime(final_df['Start Date'])


# In[26]:


final_df.info()


# In[27]:


final_df.loc[:, 'year'] = final_df['Start Date'].dt.year
final_df.loc[:, 'month'] = final_df['Start Date'].dt.month
final_df.loc[:, 'week'] = final_df['Start Date'].dt.isocalendar().week
final_df.loc[:, 'day'] = final_df['Start Date'].dt.day
final_df.loc[:, 'hour'] = final_df['Start Date'].dt.hour
final_df.loc[:, 'dayofweek'] = final_df['Start Date'].dt.dayofweek
final_df.loc[:, 'satsun'] = final_df['dayofweek'].map({0: False, 1: False, 2: False, 3: False, 4: False, 5: True, 6:True})
final_df.shape


# In[28]:


weekday_191_start = final_df.loc[(final_df['StartStation Id'] == 191) & (final_df['satsun'] == False)]
weekday_191_end = final_df.loc[(final_df['EndStation Id'] == 191) & (final_df['satsun'] == False)]
weekday_191_start = weekday_191_start.groupby('hour').count()
weekday_191_end = weekday_191_end.groupby('hour').count()
weekday_191_data = pd.DataFrame({'start_counts' : weekday_191_start.iloc[:,0],
                                'end_counts' : weekday_191_end.iloc[:,0]})

sns.lineplot(data=weekday_191_data)


# In[34]:


weekend_191_start = final_df.loc[(final_df['StartStation Id'] == 191) & (final_df['satsun'] == True)]
weekend_191_end = final_df.loc[(final_df['EndStation Id'] == 191) & (final_df['satsun'] == True)]
weekdend_191_start = weekend_191_start.groupby('hour').count()
weekend_191_end = weekend_191_end.groupby('hour').count()
weekend_191_data = pd.DataFrame({'start_counts_' : weekend_191_start.iloc[:,0],
                                'end_counts_' : weekend_191_end.iloc[:,0]})

sns.lineplot(data=weekend_191_data)


# In[29]:


# number of rental events in the months of a year 2021
import seaborn as sns
per_month_2021_group = final_df.loc[final_df['year'] == 2021].groupby('month')
month_counts = per_month_2021_group.count()

sns.barplot(x=month_counts.index, y=month_counts.iloc[:,0])


# In[30]:


# number of rental events in the months of a year 2020
import seaborn as sns
per_month_2020_group = final_df.loc[final_df['year'] == 2020].groupby('month')
month_counts = per_month_2020_group.count()

sns.barplot(x=month_counts.index, y=month_counts.iloc[:,0])


# In[ ]:





# In[26]:


station_pair_group = df_new.groupby(['StartStation Name', 'EndStation Name'])
station_pair_count = station_pair_group.count().iloc[:,0]


# In[27]:


total_by_start_station = station_pair_count.groupby('StartStation Name').sum()
rel_weight = 100.0 * station_pair_count.div(total_by_start_station, level=0)


# In[28]:


rel_weight = rel_weight.loc[rel_weight > 1.0]


# In[29]:


stations = np.union1d(df_new['StartStation Name'].unique(), df_new['EndStation Name'].unique())

mindex_square = pd.MultiIndex.from_product([stations, stations])

rel_weight_square = rel_weight.reindex(index=mindex_square, fill_value=0.0)

rel_weight_square = rel_weight_square.unstack()

sns.heatmap(rel_weight_square.values, vmin=0.0, vmax=5.0, cmap="YlGnBu")


# In[30]:


# directed weighted graph as a DiGraph instance from the networkx library
dg = nx.DiGraph()
dg.add_nodes_from(stations)


# In[31]:


edge_weights_dict = rel_weight.to_dict()

edge_weights_data = [(key1, key2, val) for (key1, key2), val in edge_weights_dict.items()]

dg.add_weighted_edges_from(edge_weights_data)


# In[32]:


node_clusterings = nx.algorithms.cluster.clustering(dg)

sorted(node_clusterings.items(), key=lambda kv: kv[1])


# In[33]:


nx.draw(dg, with_labels = True, font_color = 'white', node_shape='s')


# In[ ]:





# In[ ]:





# In[34]:


all_locs = pd.read_csv("C:/Users/prash/Documents/cyclingdata/bike_point_locations_saved.csv")
print(all_locs.shape)
all_locs.head()


# I calculate the full set of unique routes actually made within the entire usage data, 
# so that I can then run these unique routes through my journey planner. 
# I get around 400k unique routes made - out of a total possible of around 600k (777 * 776).

# In[35]:


## Generate list of unique routes
unq_locs = df_new.loc[:,('StartStation Id',
                      'EndStation Id')]
print(unq_locs.shape)
unq_locs.drop_duplicates(inplace=True)
print(unq_locs.shape)


# In[36]:


## Merge on the lat/lons

unq_locs = unq_locs.merge(right = all_locs,
                             how = 'inner',
                             left_on = 'StartStation Id',
                             right_on = 'id')

print(unq_locs.shape)

unq_locs.drop(labels = ["id", "name"], axis=1, inplace=True)
unq_locs.rename(columns={'lat': 'StartStation lat', 'lon': 'StartStation lon', 
                            'capacity': 'StartStation capacity'},
                   inplace=True)


# In[37]:


# Merge end
unq_locs = unq_locs.merge(right = all_locs,
                             how = 'inner',
                             left_on = 'EndStation Id',
                             right_on = 'id')

unq_locs.drop(labels = ["id", "name"], axis=1, inplace=True)
unq_locs.rename(columns={'lat': 'EndStation lat', 'lon': 'EndStation lon',
                           'capacity': 'EndStation capacity'},
                   inplace=True)


print(unq_locs.shape)
unq_locs.head()


# In[38]:


unq_locs_det = unq_locs.loc[:,('StartStation Id',
                               'EndStation Id',
                               'StartStation lat',
                               'StartStation lon',
                               'EndStation lat',
                               'EndStation lon')]


# In[ ]:




