# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 15:40:52 2016

@author: jiang_y
"""

def plot_smurf(filePath, fileName, save2pdf):
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    #import ggplot
    from ggplot import *
    from matplotlib.backends.backend_pdf import PdfPages
    from IPython import get_ipython
    
    # save2pdf = 1
    
    if save2pdf:    
        get_ipython().run_line_magic('matplotlib', 'qt')        # set to plot in new window
    else:
        get_ipython().run_line_magic('matplotlib', 'inline')        # set to plot inline
    
    # Open tako log and assign to df as data frame
    # filePath = 'C:/work/YongJiang/MyWork/Tools/EDS_pie/results_PALMER_PX0Z9_DLPLSU0/XCAL-SIO/PREBI/'
    os.chdir(filePath)
    # fileName = 'TAKO_Result-Log_WX61A968A1LD_520.csv'
    df=pd.read_csv(filePath + fileName)
    
    
    # Get SN
    temp = df[df.Name=='Serial Number']
    sn = temp.Value.to_string()
    sn = sn.split('-')[1]
    
    # plot SS2D 747 curves
    filter1 = 'SS2D Mean OTRC Percent Per Squeeze'
    df1 = df[df.Name==filter1]
    # select columns
    sel_cols =['Name','Run Index','Sub-Test ID','Head No.','Zone No.','KFCI Code','TPI Code','Sqz Idx','Value']
    df1 = df1[sel_cols]
    df1.columns = ['Name','Run Index','Sub-Test ID','Head','Zone','KFCI Code','TPI Code','Sqz Idx','SFR OTRC']
    df1['SN'] = sn
    
    # Force some columns to correct data type for plotting
    for col in ['Run Index','Sub-Test ID','Head','Zone']:
        df1[col] = df1[col].astype('int')
        df1[col] = df1[col].astype('object')
        
    for col in ['SFR OTRC']:
        df1[col] = df1[col].astype('float64')
    
    if save2pdf: pdf_file = PdfPages(fileName + '.pdf')       # open pdf file
    print(ggplot(df1[df1.Head==0], aes(x='Sqz Idx', y='SFR OTRC', color='Zone')) + \
          geom_line() + facet_wrap('Sub-Test ID') + ggtitle("747 Curve by SubTest ID for head 0"))
    if save2pdf: pdf_file.savefig()  # save each figure to pdf file
    print(ggplot(df1[df1.Head==1], aes(x='Sqz Idx', y='SFR OTRC', color='Zone')) + \
          geom_line() + facet_wrap('Sub-Test ID') + ggtitle("747 Curve by SubTest ID for head 1"))
    if save2pdf: pdf_file.savefig()
    
    
    # Plot final limit check
    df3 = df[(df.Name=='Ow (dB) for Zones Tested') \
             | (df.Name=='Error Metric for Zones Tested') \
             | (df.Name=='ATI for Zones Tested') \
             | (df.Name=='OTRC (Jog DACs) for Zones Tested')]
    df3 = df3[['Name','Head No.','Zone Idx','Value']]
    df3.columns = ['Name','Head','Zone','Metrics']
    # Force some columns to correct data type for plotting
    for col in ['Head','Zone']:
        df3[col] = df3[col].astype('int')
        df3[col] = df3[col].astype('object')    
    for col in ['Metrics']:
        df3[col] = df3[col].astype('float64')
    print(ggplot(df3, aes(x='Zone', y='Metrics', color='Head')) + geom_line() + facet_wrap('Name',scales='free'))
    if save2pdf: pdf_file.savefig()
    
    
    # Plot final selection
    df2 = df[(df.Name=='Final true KFCI for all user zones') | (df.Name=='Final true KTPI for all user zones') \
             | (df.Name=='Final KFCI Code Selection for All User Zones') | (df.Name=='Final TPI Code Selection for All User Zones')]
    df2 = df2[['Name','Head Idx','Zone Idx','Value']]
    df2.columns = ['Name','Head','Zone','KFCI_True']
    # Force some columns to correct data type for plotting
    for col in ['Head','Zone']:
        df2[col] = df2[col].astype('int')
        df2[col] = df2[col].astype('object')    
    for col in ['KFCI_True']:
        df2[col] = df2[col].astype('float64')
        
    print(ggplot(df2, aes(x='Zone', y='KFCI_True', color='Head')) + geom_line() + facet_wrap('Name',scales='free'))
    if save2pdf: pdf_file.savefig()
    

    
    
    if save2pdf: plt.close("all")        # close figure windows
    if save2pdf: pdf_file.close()    # close pdf file
    
    #print(ggplot(df1, aes(x='Sqz Idx', y='SFR OTRC', color='Zone')) + geom_point() + facet_wrap('Sub-Test ID'))     # print ggplot to figure
    #print(ggplot(df1, aes(x='Sqz Idx', y='SFR OTRC', color='Zone')) + geom_line() + facet_grid('Head','Sub-Test ID',scales='free_y'))
    
