"""
@author: Nicole Pililyan 
        (and Ryan Miller, code taken and tweaked from Project 2)

This file is used for analyzing the gathered data in comparison to the generated data
 and creating plots for the two data sets (self-checkout and cashier checkout)
"""

import matplotlib.pyplot as plt
import numpy as np
from processData import process_data
from scipy import stats

dt = 1 # 0 for Sel-c
gaData, gaDataSize = process_data("data",0)

dataDesc = "Checkout Times"
numBins = 35
dist_shape = 0
dist_scale = 0

if dt == 0:
    dataDesc = "Self-Checkout Times"
    dist_shape = 0.361
    dist_scale = np.log(179.998)
if dt == 1:
    dataDesc = "Cashier Checkout Times"
    dist_shape = 0.3816
    dist_scale = np.log(168.2403)
    numBins = 20


ivs = [0.99,0.95]
for i in ivs:
    # s,e = stats.lognorm.interval(i, dist_shape, scale=dist_scale)
    # s,e = stats.norm.interval(i, loc=np.mean(gaData), scale=stats.sem(gaData))
    if dt == 0:
        s,e = stats.norm.interval(i, loc=np.mean(gaData), scale=stats.sem(gaData))
        print(f"[SelfC GA] - Confidence Interval of {i}: [{s}, {e}]")
        s,e = stats.norm.interval(i, loc=214.70201500383774, scale=11.89195016569566)
        print(f"[SelfC SIM] - Confidence Interval of {i}: [{s}, {e}]")
    if dt == 1:
        s,e = stats.norm.interval(i, loc=np.mean(gaData), scale=stats.sem(gaData))
        print(f"[CashierC GA] - Confidence Interval of {i}: [{s}, {e}]")
        s,e = stats.norm.interval(i, loc=187.16508985066181, scale=11.366328204589072)
        print(f"[CashierC SIM] - Confidence Interval of {i}: [{s}, {e}]")

# sampleMean = np.mean(gaData)
# sampleSTD = np.std(gaData)
# print('Gathered Data:')
# print(f'sample mean {sampleMean:0.3f}')
# print(f'sample std {np.sqrt(sampleSTD):0.3f}')

# sampleMean = np.mean(simData)
# sampleSTD = np.std(simData)
# print('Simulation Data:')
# print(f'sample mean {sampleMean:0.3f}')
# print(f'sample std {np.sqrt(sampleSTD):0.3f}')

# print(gaDataSize)
# print(simDataSize)

# mean = np.mean(simData)
# for i,j in enumerate(simData):
#     if j > 300 or j < 0:
#         simData[i] = mean

# plt.subplot(1,2,1)
# #plt.figure()
# plt.hist(gaData, numBins, density=True, align='mid')
# plt.xlabel("Data Value")
# plt.ylabel("Probability Density")
# plt.title(f"Binned Gathered {dataDesc} Data")

# plt.subplot(1,2,2)
# #plt.figure()
# plt.hist(simData, numBins, density=True, align='mid')
# plt.xlabel("Data Value")
# plt.ylabel("Probability Density")
# plt.title(f"Binned {dataDesc} Data from Simulation")



plt.show()