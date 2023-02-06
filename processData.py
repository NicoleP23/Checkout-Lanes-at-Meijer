"""
@author: Nicole Pililyan

This file processes the collected data
"""

# imports
import pandas as pd
import numpy as np
from os import listdir
import matplotlib.pyplot as plt
from scipy import stats

# to_seconds - method to convert mm:ss string to seconds
# Parameter: timestr - the given time in minutes and seconds to convert to seconds
# Returns: the converted time of only seconds
def to_seconds(timestr):
    mm, ss = timestr.split(':')
    return int(mm) * 60 + int(ss)

# remove 'nan's from data
def remove_nan(data_arr):
    for x in data_arr:
        if pd.isna(x):
            # print("Removed")
            data_arr.remove(x)
            
# process_data - process the collected data
# folder_name - the collected data folder to process (collected data or simulation data)
def process_data(folder_name, num):
    timeline = []
    times = []
    mPerks = []
    assist = []
    waitAssist = []
    getAssist = []
    
    plot_data = False
    sampleDataSelfC = times
    sampleDataCashC = times
    
    files = listdir(f"{folder_name}")
    
    for file in files:
        # print(file)
        name, ext = file.split('.')
        df = None
        # reset for next file
        times = []
        mPerks = []
        assist = []
        waitAssist = []
        getAssist = []

        if ext == "csv":
            df = pd.read_csv(f"{folder_name}/{file}",dtype=str)
            timeline.append(f"Got {file}")
        
        if folder_name == "data":
            # convert time in first column to seconds
            for index,row in df.iterrows():
                df.iloc[index] = df.iloc[index].replace(to_replace=row[0], value=to_seconds(row[0]))
                # print(df.iloc[index])
                times.append(row[0])
                mPerks.append(row[1])
                if file == "SelfCheckoutCD.csv":
                    assist.append(row[2])
                    waitAssist.append(row[3])
                    getAssist.append(row[4])
            timeline.append(times)
            # change from strings to ints
            mPerks = [int(x) for x in mPerks]
            if file == "SelfCheckoutCD.csv":
                assist = [int(x) for x in assist]
                # print(f"Waiting: {waitAssist}\nGetting: {getAssist}")
                
                # for some reason, it wouldn't remove every nan, 
                # so I had to go through them 3 times to remove all of the 'nan's
                remove_nan(waitAssist)
                remove_nan(getAssist)
                remove_nan(waitAssist)
                remove_nan(getAssist)
                remove_nan(waitAssist)
                remove_nan(getAssist)
                
                waitAssist = [int(x) for x in waitAssist]
                getAssist = [int(x) for x in getAssist]
                timeline.append(f"Waiting: {waitAssist}\nGetting: {getAssist}")
            # print(len(mPerks))
            # print(mPerks)
            
            
            # find Means and std
            sampleMean = np.mean(times)
            sampleSTD = np.std(times)
            sampleSEM = stats.sem(times)
            samplemPerksAvg = np.mean(mPerks)            
           
            timeline.append(f'{file} Data:')
            timeline.append(f'sample mean {sampleMean:0.3f}')
            timeline.append(f'sample SEM {sampleSEM:0.3f}')
            timeline.append(f'sample std {np.sqrt(sampleSTD):0.3f}\n')
            timeline.append(f"Mean of use of mPerks: {samplemPerksAvg}")
            if file == "SelfCheckoutCD.csv":
                sampleAssistAvg = np.mean(assist)
                timeline.append(f"Mean of needing Assistance: {sampleAssistAvg}")
                timeline.append(f"Mean of waiting for Assistance: {np.mean(waitAssist)}")
                timeline.append(f"Mean of getting Assistance: {np.mean(getAssist)}\n")
                
                
            the_data = False
            bins = 20
            type = "Cashier Checkout"
            if file == "SelfCheckoutCD.csv":
                bins = 35
                type = "Self-Checkout"
            if the_data:
                plt.figure() 
                plt.bar(range(len(times)), times)
                plt.ylabel('Checkout Time') 
                plt.xlabel('Collected Index') 
                plt.title(f'Collected Times - {type}')
                
                plt.figure() 
                plt.hist(times, bins) 
                plt.xlabel('Checkout Time') 
                plt.ylabel('Frequency') 
                plt.title(f'Collected Times - {type}') 
                
            if plot_data:
                # plot data
                plt.figure()
                plt.bar(range(len(times)), times)
                plt.xlabel("Data Index")
                plt.ylabel("Data Value")
                plt.title(f"{file} Data")
        
                # plot a histogram
                plt.figure()
                numBins = 20
                if file == "SelfCheckoutCD.csv":
                    numBins = 35
                plt.hist(times, numBins, density=True, align='mid')
                plt.xlabel("Data Value")
                plt.ylabel("Probability Density")
                plt.title(f"Binned {file} Data")
                
                
                #generating plots of data 
                binSize = [5, 10, 20, 35] 
                fig, axs = plt.subplots(len(binSize), 1) 
                plt.subplots_adjust(hspace=.5) 
                titleLabel = f'{file} Num bins: ' + ' '.join([str(bs) for bs in binSize]) 
                fig.suptitle(titleLabel) 
                for idx, ax in enumerate(axs): 
                    ax.hist(times, bins=binSize[idx]) 
                
                
                plt.show() 
                # observed
                binEdges = np.linspace(0.0, np.max(times), numBins) 
                observed, _ = np.histogram(times, bins=binEdges) 
                
        # return self checkout data if num == 0
            if file == "SelfCheckoutCD.csv":
                # print(f"Got {file} at {num}")
                sampleDataSelfC = times
                # print(sampleDataSelfC)
            # return cashier checokut data if num == 1
            if file == "CashierCheckoutCD.csv":
                # print(f"Got {file} at {num}")
                sampleDataCashC = times
                # print(sampleDataCashC)
        


    # return self checkout or cashier checkout based on what is asked for
    if num == 0:
        sampleData = sampleDataSelfC
        sampleSize = len(sampleDataSelfC)
    if num == 1:
        sampleData = sampleDataCashC
        sampleSize = len(sampleDataCashC)
    
    # used to check if the correct data was being returned
    # print(f"\nReturning: {sampleData}") 
    
    printt = False
    if printt:
        for string in timeline:
            print(string)
    
    return sampleData, sampleSize 
                
        
            
process_data("data", 0)
# process_data("data", 1)

"""
Self Checkout Data uses 35 bins
Self Checkout Mean: 191.216
Self Checkout std: 7.884
Self Checkout SEM: 10.360
Self Checkout mPerks Mean: .513513
Self Checkout Assistance Mean: 0.486486
Mean of waiting for Assistance: 21.166666
Mean of getting Assistance: 20.5

Cashier Checkout Data uses 20 bins
Cashier Checkout Mean: 180.714
Cashier Checkout std: 8.254
Cashier Checkout SEM: 11.684
Cashier Checkout mPerks Mean: .428571
"""