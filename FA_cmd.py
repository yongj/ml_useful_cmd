# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 11:47:51 2016

@author: jiang_y
"""
############################### Important Modules #######################
import pandas as pd
import matplotlib.pyplot as plt
from ggplot import *
from matplotlib.backends.backend_pdf import PdfPages

import os
from os import listdir


# system operations
import os
os.getcwd()
os.chdir(mypath)

from os import listdir
filenames = listdir(mypath)
for aa in filenames:
    xxx

############################### Data Preprocess ############################
# Load data: read data from csv
import pandas as pd
mypath = 'C:/work/YongJiang/MyWork/Tools/EDS_pie/results_PALMER_PX0Z9_DLPLSU0/XCAL-SIO/PREBI/'
fileName = 'TAKO_Result-Log_WX61A968A1LD_520.csv'
df=pd.read_csv(mypath + fileName)

# write data to csv
df.to_csv('ss2d.csv')

# filtering
filter1 = 'SS2D Mean OTRC Percent Per Squeeze'
df1 = df[df.Name==filter1]

# select columns
sel_cols =['Name','Run Index','Sub-Test ID','Head No.','Zone No.','KFCI Code','TPI Code','Sqz Idx','Value']
df2 = df1[sel_cols]
df2.columns = ['Name','Run Index','Sub-Test ID','Head','Zone','KFCI Code','TPI Code','Sqz Idx','SFR OTRC']

sn = fileName.split('_')[2]
df2['SN'] = sn


# filtering
temp = df2[(df2.Head == 0) & (df2.Zone == 2) & (df2['Sub-Test ID'] == 3)]
           
# view column names
df2.columns
# veiw column data types
df2.dtypes  # the data types of all columns
df2.Name.dtype # the type of a specific column
df2.Head = df2.Head.astype('int')
df2.Head = df2.Head.astype('object')

col = 'Sub-Test ID'
df2[col] = df2[col].astype('int')
df2[col] = df2[col].astype('object')



df2['Zone'].hist(bins=60)
df2['KFCI Code'].hist(bins=51)
df2['TPI Code'].hist(bins=41)



# plot
import matplotlib.pyplot as plt

# plot in new window or inline
%matplotlib qt
%matplotlib inline

from matplotlib.backends.backend_pdf import PdfPages
pp = PdfPages("test1.pdf")
pp.savefig()
pp.close()

plt.plot(x,y)
#plt.plot(temp['Sqz Idx'],temp['SFR OTRC'])
temp['Sqz Idx']=temp['Sqz Idx'].astype(float)
temp['SFR OTRC']=temp['SFR OTRC'].astype(float)
temp.plot(x='Sqz Idx',y='SFR OTRC')

temp.plot(x='Sqz Idx',y='SFR OTRC',color='Head')

# ggplot
from ggplot import *
ggplot(diamonds, aes(x='carat', y='price', color='cut')) +\
    geom_point() +\
    scale_color_brewer(type='diverging', palette=4) +\
    xlab("Carats") + ylab("Price") + ggtitle("Diamonds")

ggplot(df2, aes(x='Sqz Idx', y='SFR OTRC', color='Sub-Test ID')) +\
    geom_point() +\
    scale_color_brewer(type='qual', palette=2) +\
    xlab("Sqz Idx") + ylab("SFR OTRC") + ggtitle("747 Curve")    
    
ggplot(df2, aes(x='Sqz Idx', y='SFR OTRC', color='Sub-Test ID')) +\
    geom_point() + facet_grid('Head','Zone',scales='free_y') + \
    scale_color_brewer(type='qual', palette=2) +\
    xlab("Sqz Idx") + ylab("SFR OTRC") + ggtitle("747 Curve")  

# theme    
ggplot(df2, aes(x='Sqz Idx', y='SFR OTRC', color='Sub-Test ID')) +\
    geom_point() + facet_grid('Head','Zone',scales='free_y') + \
    scale_color_brewer(type='qual', palette=2) +\
    xlab("Sqz Idx") + ylab("SFR OTRC") + ggtitle("747 Curve") + theme_bw()
    
ggplot(df2, aes(x='Sqz Idx', y='SFR OTRC', color='Sub-Test ID')) +\
    geom_point() + facet_wrap('Head') + \
    scale_color_brewer(type='qual', palette=2) +\
    xlab("Sqz Idx") + ylab("SFR OTRC") + ggtitle("747 Curve")  

# save plot to pdf    
p = ggplot(df2, aes(x='Sqz Idx', y='SFR OTRC', color='Sub-Test ID')) +\
    geom_point() + facet_grid('Head','Zone',scales='free_y') + \
    scale_color_brewer(type='qual', palette=2) +\
    xlab("Sqz Idx") + ylab("SFR OTRC") + ggtitle("747 Curve") 
ggsave(p,'test.png')
    




import pandas as pd

mypath = 'C:/work/YongJiang/MyWork/Tools/EDS_pie/results_PALMER_PX0Z9_DLPLSU0/XCAL-SIO/PREBI/'
os.getcwd()
os.chdir(mypath)

filenames = listdir(mypath)

df=pd.read_csv(mypath + filenames[3])

print(df.columns)





filter1 = 'SS2D Mean OTRC Percent Per Squeeze'
filter1 = 'SFR S2D Percent'
data = pd.DataFrame()

for aa in filenames:
    df=pd.read_csv(mypath + aa)
    df1 = df[df.Name==filter1]
    sel_cols =['Name','Run Index','Sub-Test ID','Head No.','Zone No.','KFCI Code','TPI Code','Sqz Idx','Value']
    df1 = df1[sel_cols]
    df1.columns = ['Name','Run Index','Sub-Test ID','Head','Zone','KFCI Code','TPI Code','Sqz Idx','SFR OTRC']
    sn = aa.split('_')[2]
    df1['SN'] = sn
    data = pd.concat([data,df1])
    
data.to_csv('ss2d.csv')