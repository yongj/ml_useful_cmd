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
    import seaborn as sns
    
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
    
    
    # Force some columns to correct data type for plotting
    for col in ['Run Index','Sub-Test ID','Head No.','Phys. Head No.','Zone No.','TPI Code','KFCI Code',\
    'Zone Idx','Cap Idx','Head','Zone','TPI Idx','KFCI Idx','Kfci Idx','Head Idx','Sqz Idx','Offtrk Idx']:
        #df[col] = df[col].astype('int')
        df[col] = df[col].astype('object')
    
    # Get SN
    temp = df[df.Name=='Serial Number']
    sn = temp.Value.to_string()
    sn = sn.split('-')[1]
    
    ############### Plot 747 curve ####################
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
    
    
    ############### Plot final limit check ##############
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
    print(ggplot(df3, aes(x='Zone', y='Metrics', color='Head')) + geom_line() + facet_wrap('Name',scales='free') + \
          ggtitle("Final Limit Check"))
    if save2pdf: pdf_file.savefig()
    
    
    ############### Plot final selection ###############
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
        
    print(ggplot(df2, aes(x='Zone', y='KFCI_True', color='Head')) + geom_line() + facet_wrap('Name',scales='free') + \
          ggtitle("Final KFCI / TPI Selection"))
    if save2pdf: pdf_file.savefig()
    

    ############### Plot KFCI true value at different subtest ##############
    
    df3 = df[df.Name=='Final Selected True KFCI']
    # select columns
    sel_cols =['Name','Sub-Test ID','Head','Zone','Value']
    df3 = df3[sel_cols]
    df3.columns = ['Name','Sub-Test ID','Head','Zone','KFCI_True']
    df3['SN'] = sn
    
    for col in ['Sub-Test ID','Head','Zone']:
        df3[col] = df3[col].astype('int')
        df3[col] = df3[col].astype('object')
    df3['KFCI_True'] = df3['KFCI_True'].astype('float64')
        

    df3_1 = df[df.Name=='Final true KFCI for all user zones']
    # select columns
    sel_cols =['Name','Sub-Test ID','Head Idx','Zone Idx','Value']
    df3_1 = df3_1[sel_cols]
    df3_1.columns = ['Name','Sub-Test ID','Head','Zone','KFCI_True']
    df3_1['SN'] = sn
    df3_1['Sub-Test ID'] = 'Final_Selection'
    
    for col in ['Head','Zone']:
        df3[col] = df3[col].astype('int')
        df3[col] = df3[col].astype('object')
    df3_1['KFCI_True'] = df3_1['KFCI_True'].astype('float64')
    
    df3 = pd.concat([df3,df3_1])
    
    print(ggplot(df3, aes(x='Zone', y='KFCI_True', color='Sub-Test ID')) + geom_line() + facet_wrap('Head',scales='free') + \
          ggtitle("True KFCI Selections in Different Subtest"))
    # print(sns.factorplot(data=df3, x="Zone", y="KFCI_True", col="Head", hue="Sub-Test ID", palette="muted",size=8))
    if save2pdf: pdf_file.savefig()

    ############### Plot KFCI code at different subtest ##############
    
    df3 = df[df.Name=='Final Selected KFCI Code']
    df3 = df3[(df3['Sub-Test ID']==2) | (df3['Sub-Test ID']==10)]
    # select columns
    sel_cols =['Name','Sub-Test ID','Head','Zone','Value']
    df3 = df3[sel_cols]
    df3.columns = ['Name','Sub-Test ID','Head','Zone','KFCI_Code']
    df3['SN'] = sn
    
    for col in ['Sub-Test ID','Head','Zone']:
        df3[col] = df3[col].astype('int')
        df3[col] = df3[col].astype('object')
    df3['KFCI_Code'] = df3['KFCI_Code'].astype('float64')
        

    df3_1 = df[df.Name=='Final KFCI Code Selection for All User Zones']
    # select columns
    sel_cols =['Name','Sub-Test ID','Head Idx','Zone Idx','Value']
    df3_1 = df3_1[sel_cols]
    df3_1.columns = ['Name','Sub-Test ID','Head','Zone','KFCI_Code']
    df3_1['SN'] = sn
    df3_1['Sub-Test ID'] = 'Final_Selection'
    
    for col in ['Head','Zone']:
        df3[col] = df3[col].astype('int')
        df3[col] = df3[col].astype('object')
    df3_1['KFCI_Code'] = df3_1['KFCI_Code'].astype('float64')
    
    df3 = pd.concat([df3,df3_1])
    
    print(ggplot(df3, aes(x='Zone', y='KFCI_Code', color='Sub-Test ID')) + geom_line() + facet_wrap('Head',scales='free') + \
          ggtitle("KFCI Code Selections in Different Subtest"))
    # print(sns.factorplot(data=df3, x="Zone", y="KFCI_True", col="Head", hue="Sub-Test ID", palette="muted",size=8))
    if save2pdf: pdf_file.savefig()

    ############### Plot TPI true value at different subtest ##############
    
    df3 = df[df.Name=='Initial TPI adjustment selected TRUE KTPI']
    df3 = df3[df3['Sub-Test ID']==7]
    # select columns
    sel_cols =['Name','Sub-Test ID','Head Idx','Zone Idx','Value']
    df3 = df3[sel_cols]
    df3.columns = ['Name','Sub-Test ID','Head','Zone','TPI_True']
    df3['SN'] = sn
    
    for col in ['Sub-Test ID','Head','Zone']:
        df3[col] = df3[col].astype('int')
        df3[col] = df3[col].astype('object')
    df3['TPI_True'] = df3['TPI_True'].astype('float64')
        

    df3_1 = df[df.Name=='Final true KTPI for all user zones']
    # select columns
    sel_cols =['Name','Sub-Test ID','Head Idx','Zone Idx','Value']
    df3_1 = df3_1[sel_cols]
    df3_1.columns = ['Name','Sub-Test ID','Head','Zone','TPI_True']
    df3_1['SN'] = sn
    df3_1['Sub-Test ID'] = 'Final_Selection'
    
    for col in ['Head','Zone']:
        df3[col] = df3[col].astype('int')
        df3[col] = df3[col].astype('object')
    df3_1['TPI_True'] = df3_1['TPI_True'].astype('float64')
    
    df3 = pd.concat([df3,df3_1])
    
    print(ggplot(df3, aes(x='Zone', y='TPI_True', color='Sub-Test ID')) + geom_line() + facet_wrap('Head',scales='free') + \
          ggtitle("True TPI Selections in Different Subtest"))
    # print(sns.factorplot(data=df3, x="Zone", y="KFCI_True", col="Head", hue="Sub-Test ID", palette="muted",size=8))
    if save2pdf: pdf_file.savefig()

    ############### Plot TPI code at different subtest ##############
    
    df3 = df[df.Name=='Initial TPI Code Selection for All Heads and User Zones']
    df3 = df3[df3['Sub-Test ID']==7]
    # select columns
    sel_cols =['Name','Sub-Test ID','Head Idx','Zone Idx','Value']
    df3 = df3[sel_cols]
    df3.columns = ['Name','Sub-Test ID','Head','Zone','TPI_Code']
    df3['SN'] = sn
    
    for col in ['Sub-Test ID','Head','Zone']:
        df3[col] = df3[col].astype('int')
        df3[col] = df3[col].astype('object')
    df3['TPI_Code'] = df3['TPI_Code'].astype('float64')
        

    df3_1 = df[df.Name=='Final TPI Code Selection for All User Zones']
    # select columns
    sel_cols =['Name','Sub-Test ID','Head Idx','Zone Idx','Value']
    df3_1 = df3_1[sel_cols]
    df3_1.columns = ['Name','Sub-Test ID','Head','Zone','TPI_Code']
    df3_1['SN'] = sn
    df3_1['Sub-Test ID'] = 'Final_Selection'
    
    for col in ['Head','Zone']:
        df3[col] = df3[col].astype('int')
        df3[col] = df3[col].astype('object')
    df3_1['TPI_Code'] = df3_1['TPI_Code'].astype('float64')
    
    df3 = pd.concat([df3,df3_1])
    
    print(ggplot(df3, aes(x='Zone', y='TPI_Code', color='Sub-Test ID')) + geom_line() + facet_wrap('Head',scales='free') + \
          ggtitle("TPI Code Selections in Different Subtest"))
    # print(sns.factorplot(data=df3, x="Zone", y="KFCI_True", col="Head", hue="Sub-Test ID", palette="muted",size=8))
    if save2pdf: pdf_file.savefig()

    
    ###################### Close figures and pdf file ###################
    if save2pdf: plt.close("all")        # close figure windows
    if save2pdf: pdf_file.close()    # close pdf file
    
    #print(ggplot(df1, aes(x='Sqz Idx', y='SFR OTRC', color='Zone')) + geom_point() + facet_wrap('Sub-Test ID'))     # print ggplot to figure
    #print(ggplot(df1, aes(x='Sqz Idx', y='SFR OTRC', color='Zone')) + geom_line() + facet_grid('Head','Sub-Test ID',scales='free_y'))
    

    
    
def pull_kfci_tpi(filePath, fileName):
    import os
    import pandas as pd

    # Open tako log and assign to df as data frame
    # filePath = 'C:/work/YongJiang/MyWork/Tools/EDS_pie/results_PALMER_PX0Z9_DLPLSU0/XCAL-SIO/PREBI/'
    os.chdir(filePath)
    # fileName = 'TAKO_Result-Log_WX61A968A1LD_520.csv'
    df=pd.read_csv(filePath + fileName)
    
    # Get SN
    temp = df[df.Name=='Serial Number']
    sn = temp.Value.to_string()
    sn = sn.split('-')[1]

    df1 = pd.DataFrame()
    
    ############### Plot KFCI true value at different subtest ##############
    
    df3 = df[df.Name=='Final Selected True KFCI']
    # select columns
    sel_cols =['Name','Sub-Test ID','Head','Zone','Value']
    df3 = df3[sel_cols]
    df3.columns = ['Name','Sub-Test ID','Head','Zone','Value']
    df3['SN'] = sn
    
    df1 = pd.concat([df1,df3])

    df3_1 = df[df.Name=='Final true KFCI for all user zones']
    # select columns
    sel_cols =['Name','Sub-Test ID','Head Idx','Zone Idx','Value']
    df3_1 = df3_1[sel_cols]
    df3_1.columns = ['Name','Sub-Test ID','Head','Zone','Value']
    df3_1['SN'] = sn
    df3_1['Sub-Test ID'] = 'Final_Selection'
    

    
    df1 = pd.concat([df1,df3_1])
    


    ############### Plot KFCI code at different subtest ##############
    
    df3 = df[df.Name=='Final Selected KFCI Code']
    df3 = df3[(df3['Sub-Test ID']==2) | (df3['Sub-Test ID']==10)]
    # select columns
    sel_cols =['Name','Sub-Test ID','Head','Zone','Value']
    df3 = df3[sel_cols]
    df3.columns = ['Name','Sub-Test ID','Head','Zone','Value']
    df3['SN'] = sn
    
    df1 = pd.concat([df1,df3])
        

    df3_1 = df[df.Name=='Final KFCI Code Selection for All User Zones']
    # select columns
    sel_cols =['Name','Sub-Test ID','Head Idx','Zone Idx','Value']
    df3_1 = df3_1[sel_cols]
    df3_1.columns = ['Name','Sub-Test ID','Head','Zone','Value']
    df3_1['SN'] = sn
    df3_1['Sub-Test ID'] = 'Final_Selection'
    

    
    df1 = pd.concat([df1,df3_1])
    


    ############### Plot TPI true value at different subtest ##############
    
    df3 = df[df.Name=='Initial TPI adjustment selected TRUE KTPI']
    df3 = df3[df3['Sub-Test ID']==7]
    # select columns
    sel_cols =['Name','Sub-Test ID','Head Idx','Zone Idx','Value']
    df3 = df3[sel_cols]
    df3.columns = ['Name','Sub-Test ID','Head','Zone','Value']
    df3['SN'] = sn
    df1 = pd.concat([df1,df3])
    

        

    df3_1 = df[df.Name=='Final true KTPI for all user zones']
    # select columns
    sel_cols =['Name','Sub-Test ID','Head Idx','Zone Idx','Value']
    df3_1 = df3_1[sel_cols]
    df3_1.columns = ['Name','Sub-Test ID','Head','Zone','Value']
    df3_1['SN'] = sn
    df3_1['Sub-Test ID'] = 'Final_Selection'
    

    
    df1 = pd.concat([df1,df3_1])
    
 
    ############### Plot TPI code at different subtest ##############
    
    df3 = df[df.Name=='Initial TPI Code Selection for All Heads and User Zones']
    df3 = df3[df3['Sub-Test ID']==7]
    # select columns
    sel_cols =['Name','Sub-Test ID','Head Idx','Zone Idx','Value']
    df3 = df3[sel_cols]
    df3.columns = ['Name','Sub-Test ID','Head','Zone','Value']
    df3['SN'] = sn
    df1 = pd.concat([df1,df3])
  

    df3_1 = df[df.Name=='Final TPI Code Selection for All User Zones']
    # select columns
    sel_cols =['Name','Sub-Test ID','Head Idx','Zone Idx','Value']
    df3_1 = df3_1[sel_cols]
    df3_1.columns = ['Name','Sub-Test ID','Head','Zone','Value']
    df3_1['SN'] = sn
    df3_1['Sub-Test ID'] = 'Final_Selection'
    
  
    df1 = pd.concat([df1,df3_1])
    
    return df1
    


 