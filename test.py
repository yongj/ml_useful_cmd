# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 17:40:54 2016

@author: jiang_y
"""

#%%
import os
os.chdir('C:/Users/jiang_y/Documents/MachineLearning/spyder')

from fun_plot_smurf import plot_smurf, pull_kfci_tpi

filePath = 'C:/work/YongJiang/MyWork/Tools/EDS_pie/results_PALMER_PX0Z9_DLPLSU0/XCAL-SIO/PREBI/'
fileName = 'TAKO_Result-Log_WX61A968A1LD_520.csv'
#%%
plot_smurf(filePath, fileName, save2pdf=1)

#%%
df = pull_kfci_tpi(filePath, fileName)
#%%


import pandas as pd
os.chdir(filePath)
filenames = os.listdir(filePath)

data = pd.DataFrame()

for aa in filenames:
    df = pull_kfci_tpi(filePath, aa)
    data = pd.concat([data,df])
    
data.to_csv('kfci_tpi.csv')