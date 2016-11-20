# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 09:26:30 2016

@author: jiang_y
"""
# ref: http://www.gregreda.com/2013/10/26/intro-to-pandas-data-structures/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.set_option('max_columns', 50)
%matplotlib inline

############################## Create series #################################
# Build series from list or dictionary
s = pd.Series([7, 'Heisenberg', 3.14, -1789710578, 'Happy Eating!'])
s = pd.Series([7, 'Heisenberg', 3.14, -1789710578, 'Happy Eating!'],
              index=['A', 'Z', 'C', 'Y', 'E'])

d = {'Chicago': 1000, 'New York': 1300, 'Portland': 900, 'San Francisco': 1100,
     'Austin': 450, 'Boston': None}
cities = pd.Series(d)

############################## Work with series ##############################
# Select rows
cities['Chicago']       # by index
cities[['Chicago', 'Portland', 'San Francisco']]        # or a list of index
cities[cities < 1000]   # boolean index
cities[0:3]     # by range
cities[:3]
cities[3:]

############################# Create data frames ##############################
# Build data frames from dictionary of list
data = {'year': [2010, 2011, 2012, 2011, 2012, 2010, 2011, 2012],
        'team': ['Bears', 'Bears', 'Bears', 'Packers', 'Packers', 'Lions', 'Lions', 'Lions'],
        'wins': [11, 8, 10, 15, 11, 6, 10, 4],
        'losses': [5, 8, 6, 1, 5, 10, 6, 12]}
football = pd.DataFrame(data)       # By default, the columns will be ordered alphabetically
football = pd.DataFrame(data, columns=['year', 'team', 'wins', 'losses'])   # Specify the order by "columns" parameter

# read from files
from_csv = pd.read_csv('mariano-rivera.csv')
my_dataframe.to_csv('path_to_file.csv')

# excel. ref: http://pandas.pydata.org/pandas-docs/stable/io.html#io-excel-reader
football = pd.read_excel('football.xlsx', 'Sheet1')
df.to_excel('path_to_file.xlsx', sheet_name='Sheet1')

# clipborad. ref: http://pandas.pydata.org/pandas-docs/stable/io.html#io-clipboard
clipdf = pd.read_clipboard()
df.to_clipboard()

# read from database
from pandas.io import sql
import sqlite3

conn = sqlite3.connect('/Users/gjreda/Dropbox/gregreda.com/_code/towed')
query = "SELECT * FROM towed WHERE make = 'FORD';"

results = sql.read_sql(query, con=conn)
results.head()

# read from URL
url = 'https://raw.github.com/gjreda/best-sandwiches/master/data/best-sandwiches-geocode.tsv'

# fetch the text from the URL and read it into a DataFrame
from_url = pd.read_table(url, sep='\t')
from_url.head(3)

################################## Work with data frames #######################################
# overview
df2.info()
df2.describe()
df2.head()
df2.tail()
df2.dtypes
df2.shape

# slicing
df2[2:6]
df2[2:]
df2[:6]

users.reset_index(inplace=True) # reset index. may be needed when joining

# select multiple rows: Use loc for label-based indexing and iloc for positional indexing
df2.iloc[1:5]
df2.iloc[[1,3,5]]

# selecting columns
df2.Head
df2['Head']     # data type will be series
df2[['Head','Zone']]        # data type will be data frame

df2[df2.Head==1]
df2[(df2.Head==1) & (df2.Zone==2)]

# merging. see examples here: http://pandas.pydata.org/pandas-docs/stable/merging.html     
# concatenate along rows (vertically)
data = pd.DataFrame()
for aa in filenames:
    ...
    data = pd.concat([data,df])
    
# concatenate along columns (horizontally)    
pd.concat([left_frame, right_frame], axis=1)

# we can also use append methon to join vertically
result = df1.append(df2)
result = df1.append([df2, df3])

# join with multiple keys
left = pd.DataFrame({'key1': ['K0', 'K0', 'K1', 'K2'],
                     'key2': ['K0', 'K1', 'K0', 'K1'],
                     'A': ['A0', 'A1', 'A2', 'A3'],
                     'B': ['B0', 'B1', 'B2', 'B3']})

right = pd.DataFrame({'key1': ['K0', 'K1', 'K1', 'K2'],
                      'key2': ['K0', 'K0', 'K0', 'K0'],
                      'C': ['C0', 'C1', 'C2', 'C3'],
                      'D': ['D0', 'D1', 'D2', 'D3']})
result = pd.merge(left, right, on=['key1', 'key2'])

result = pd.merge(left, right, how='left', on=['key1', 'key2'])
result = pd.merge(left, right, how='right', on=['key1', 'key2'])
result = pd.merge(left, right, how='outer', on=['key1', 'key2'])
result = pd.merge(left, right, how='inner', on=['key1', 'key2'])

# Grouping (like pivot table in excel)
headers = ['name', 'title', 'department', 'salary']
chicago = pd.read_csv('city-of-chicago-salaries.csv')
by_dept = chicago.groupby('department')
by_dept.count()
by_dept.size()
by_dept.sum()
by_dept.mean()
by_dept.median()

################################### Plotting in pandas ########################