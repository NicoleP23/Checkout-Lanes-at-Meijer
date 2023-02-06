"""
@author: Nicole Pililyan 
        (and Ryan Miller, code taken and tweaked from Project 2)

This file checks the fit of the data and helps to figure out which distribution
to go with for the two collected data sets (self-checkout and cashier checkout)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from processData import process_data
# import sys
import os

def chi2_uniform(sd,ss,nb):
    print("Starting Uniform fit...")
    plt.figure()
    binEdges = np.linspace(0.0, np.max(sd), nb)
    observed, _ = np.histogram(sd, bins=binEdges)
    

    # MLE 
        # to get Chi-Squared to not be 'nan', .fit must have "floc" as a variable, not loc
        # However, to get the uniform fit that seemed most correct, had to have the variable be "loc"
    fit_loc, fit_scale=stats.uniform.fit(sd, floc=0, scale=1)

    print(f"Uniform: Alpha: {fit_loc} Beta: {fit_scale}")
    
    # don't do these two lines if want .---- as the graph, rather than a hill
    fit_scale = fit_scale - fit_loc # the scale given is the range, not the actual upper boundary
    print(f"Scale is: {fit_scale}")
    
    # expected
    expectedProb = stats.uniform.cdf(binEdges, scale=fit_scale, loc=fit_loc)
    # print(f"EXPECTEDPROB:\n{expectedProb}")
    expectedProb[-1] += 1.0 - np.sum(np.diff(expectedProb))  
    expected = ss * np.diff(expectedProb)
    # print(f"EXPECTEDPROB:\n{expected}")
    
    binMidPt = (binEdges[1:] + binEdges[:-1]) / 2
    plt.hist(sd, bins=binEdges, label='Observed')
    plt.plot(binMidPt, expected, 'or-', label='Expected')
    plt.plot(binMidPt, observed, 'oy-', label='Observed')
    plt.legend()
    plt.title("Uniform")
    
    # print(f"Observed: {observed}\nExpected: {expected}")
    chiSq, pValue = stats.chisquare(f_obs=observed, f_exp=expected, ddof=0)
    print(f'ChiSquare Statistic {chiSq:0.3f} P value {pValue:0.3f}')

    print('H0: (null hypothesis) Sample data follows the hypothesized distribution.')
    print('H1: (alternative hypothesis) Sample data does not follow a hypothesized distribution.')

    if pValue >= 0.05:
        print('we can not reject the null hypothesis')
    else:
        print('we reject the null hypothesis')
        
def ks_uniform(sd,ss):
    print("Starting Uniform KS fit...")
    plt.figure()

    fit_loc, fit_scale=stats.uniform.fit(sd, floc=0, scale=1)
    fit_scale = fit_scale - fit_loc # the scale given is the range, not the actual upper boundary
    print(f"Uniform: Alpha: {fit_loc} Beta: {fit_scale}")
    x = np.linspace(stats.uniform.ppf(0.01, fit_loc),stats.uniform.ppf(0.99, fit_loc), ss)

    KS_stat, p_value = stats.kstest(sd, stats.uniform.cdf(x, fit_loc))
    # arbitrary cdf tested
    sortedData = np.sort(sd)

    count = np.ones(ss)
    count = np.cumsum(count) / ss

    cdf = stats.uniform.cdf(sortedData, loc=fit_loc, scale=fit_scale)

    plt.plot(sortedData, count, 'b', label='sample data')
    plt.plot(sortedData, cdf, 'r', label='theoretical data')
    plt.legend()
    plt.xlabel('data sample')
    plt.ylabel('cummulative frequency')
    plt.title('Uniform KS')

    KS_stat, p_value = stats.ks_2samp(count, cdf)

    print('KS statistic', KS_stat)
    print('p value', p_value)

    print('H0: (null hypothesis) Sample data comes from same distribution as theoretical distribution.')
    print('H1: (alternative hypothesis) Sample data does not comes from same distribution as theoretical distribution.')

    if p_value >= 0.05:
        print('we can not reject the null hypothesis')
    else:
        print('we reject the null hypothesis')

def chi2_expon(sd,ss,nb):
    # observed
    print("Starting Exponential Fit...")
    plt.figure()
    binEdges = np.linspace(0.0, np.max(sd), nb)
    observed, _ = np.histogram(sd, bins=binEdges)

    # MLE
    fit_loc, fit_beta = stats.expon.fit(sd, floc=0)
    print(f"Exponential MLE: {fit_beta}")

    # expected
    expectedProb = stats.expon.cdf(binEdges, scale=fit_beta)
    expectedProb[-1] += 1.0 - np.sum(np.diff(expectedProb))
    expected = ss * np.diff(expectedProb)

    binMidPt = (binEdges[1:] + binEdges[:-1]) / 2
    plt.hist(sd, bins=binEdges, label='Observed')
    plt.plot(binMidPt, expected, 'or-', label='Expected')
    plt.plot(binMidPt, observed, 'oy-', label='Observed')
    plt.legend()

    chiSq, pValue = stats.chisquare(f_obs=observed, f_exp=expected)
    print(f'ChiSquare Statistic {chiSq:0.3f} P value {pValue:0.3f}')

    print('H0: (null hypothesis) Sample data follows the hypothesized distribution.')
    print('H1: (alternative hypothesis) Sample data does not follow a hypothesized distribution.')

    if pValue >= 0.05:
        print('we can not reject the null hypothesis')
    else:
        print('we reject the null hypothesis')

def ks_expon(sd,ss):
    
    print("Starting Exponential KS Fit...")
    plt.figure()

    KS_stat, p_value = stats.kstest(sd, stats.expon.cdf)
    # arbitrary cdf tested
    sortedData = np.sort(sd)
    sampleScale = np.mean(sd)
    print(f"Exponential MLE: {sampleScale}")

    count = np.ones(ss)
    count = np.cumsum(count) / ss

    cdf = stats.expon.cdf(sortedData, scale=sampleScale)

    plt.plot(sortedData, count, 'b', label='sample data')
    plt.plot(sortedData, cdf, 'r', label='theoretical data')
    plt.legend()
    plt.xlabel('data sample')
    plt.ylabel('cummulative frequency')

    KS_stat, p_value = stats.ks_2samp(count, cdf)

    print('KS statistic', KS_stat)
    print('p value', p_value)

    print('H0: (null hypothesis) Sample data comes from same distribution as theoretical distribution.')
    print('H1: (alternative hypothesis) Sample data does not comes from same distribution as theoretical distribution.')

    if p_value >= 0.05:
        print('we can not reject the null hypothesis')
    else:
        print('we reject the null hypothesis')

def ad_expon(sd):
    
    print("Starting Exponential Anderson Fit...")

    print(stats.anderson(sd,'expon'))     

def chi2_weibull(sd,ss,nb):
    # observed
    print("Starting Weibull Fit... ")
    plt.figure()
    binEdges = np.linspace(0.0, np.max(sd), nb)
    observed, _ = np.histogram(sd, bins=binEdges)


    # MLE 
    fit_alpha, fit_loc, fit_beta=stats.weibull_min.fit(sd, floc=0)

    print(f"Weibull: Alpha: {fit_alpha} Beta: {fit_beta}")

    # expected
    expectedProb = stats.weibull_min.cdf(binEdges, fit_alpha, scale=fit_beta, loc=fit_loc)
    expectedProb[-1] += 1.0 - np.sum(np.diff(expectedProb))
    expected = ss * np.diff(expectedProb)

    binMidPt = (binEdges[1:] + binEdges[:-1]) / 2
    plt.hist(sd, bins=binEdges, label='Observed')
    plt.plot(binMidPt, expected, 'or-', label='Expected')
    plt.plot(binMidPt, observed, 'oy-', label='Observed')
    plt.legend()
    plt.title("Weibull")
    # print(f"Observed: {observed}\nExpected: {expected}")

    chiSq, pValue = stats.chisquare(f_obs=observed, f_exp=expected, ddof=0)
    print(f'ChiSquare Statistic {chiSq:0.3f} P value {pValue:0.3f}')

    print('H0: (null hypothesis) Sample data follows the hypothesized distribution.')
    print('H1: (alternative hypothesis) Sample data does not follow a hypothesized distribution.')

    if pValue >= 0.05:
        print('we can not reject the null hypothesis')
    else:
        print('we reject the null hypothesis')

def ks_weibull(sd,ss):
    print("Starting Weibull KS Fit...")
    plt.figure()

    fit_alpha, fit_loc, fit_beta=stats.weibull_min.fit(sd, floc=0)
    print(f"Weibull: Alpha: {fit_alpha} Beta: {fit_beta}")
    x = np.linspace(stats.weibull_min.ppf(0.01, fit_alpha),stats.weibull_min.ppf(0.99, fit_alpha), ss)

    KS_stat, p_value = stats.kstest(sd, stats.weibull_min.cdf(x, c=fit_alpha))
    # arbitrary cdf tested
    sortedData = np.sort(sd)

    count = np.ones(ss)
    count = np.cumsum(count) / ss

    cdf = stats.weibull_min.cdf(sortedData, c=fit_alpha, loc=fit_loc, scale=fit_beta)

    plt.plot(sortedData, count, 'b', label='sample data')
    plt.plot(sortedData, cdf, 'r', label='theoretical data')
    plt.legend()
    plt.xlabel('data sample')
    plt.ylabel('cummulative frequency')
    plt.title('Weibull KS')

    KS_stat, p_value = stats.ks_2samp(count, cdf)

    print('KS statistic', KS_stat)
    print('p value', p_value)

    print('H0: (null hypothesis) Sample data comes from same distribution as theoretical distribution.')
    print('H1: (alternative hypothesis) Sample data does not comes from same distribution as theoretical distribution.')

    if p_value >= 0.05:
        print('we can not reject the null hypothesis')
    else:
        print('we reject the null hypothesis')

def chi2_lognorm(sd,ss,nb):
    # observed
    print("Starting Lognormal Fit... ")
    plt.figure()
    binEdges = np.linspace(0.0, np.max(sd), nb)
    observed, _ = np.histogram(sd, bins=binEdges)

    # MLE 
    fit_alpha, fit_loc, fit_beta=stats.lognorm.fit(sd, floc=0)
    print(f"Lognorm: Shape: {fit_alpha} Scale: {fit_beta} Location: {fit_loc}")

    # expected
    expectedProb = stats.lognorm.cdf(binEdges, fit_alpha, scale=fit_beta, loc=fit_loc)
    expectedProb[-1] += 1.0 - np.sum(np.diff(expectedProb))
    expected = ss * np.diff(expectedProb)
    
    binMidPt = (binEdges[1:] + binEdges[:-1]) / 2
    plt.hist(sd, bins=binEdges, label='Observed')
    plt.plot(binMidPt, expected, 'or-', label='Expected')
    plt.plot(binMidPt, observed, 'oy-', label='Observed')
    plt.legend()
    plt.title("Lognormal")

    chiSq, pValue = stats.chisquare(f_obs=observed, f_exp=expected)
    print(f'ChiSquare Statistic {chiSq:0.3f} P value {pValue:0.3f}')
    print('H0: (null hypothesis) Sample data follows the hypothesized distribution.')
    print('H1: (alternative hypothesis) Sample data does not follow a hypothesized distribution.')

    if pValue >= 0.05:
        print('we can not reject the null hypothesis')
    else:
        print('we reject the null hypothesis')

def ks_lognorm(sd, ss): 
    print("Starting Lognorm KS Fit...")
    plt.figure()

    fit_alpha, fit_loc, fit_beta=stats.lognorm.fit(sd, floc=0)
    print(f"Lognorm: Alpha: {fit_alpha} Beta: {fit_beta}")

    x = np.linspace(stats.lognorm.ppf(0.01, fit_alpha),stats.lognorm.ppf(0.99, fit_alpha), ss)

    KS_stat, p_value = stats.kstest(sd, stats.lognorm.cdf(x, s=fit_alpha))
    # arbitrary cdf tested
    sortedData = np.sort(sd)

    count = np.ones(ss)
    count = np.cumsum(count) / ss

    cdf = stats.lognorm.cdf(sortedData, s=fit_alpha, loc=fit_loc, scale=fit_beta)

    plt.plot(sortedData, count, 'b', label='sample data')
    plt.plot(sortedData, cdf, 'r', label='theoretical data')
    plt.legend()
    plt.xlabel('data sample')
    plt.ylabel('cummulative frequency')

    KS_stat, p_value = stats.ks_2samp(count, cdf)

    print('KS statistic', KS_stat)
    print('p value', p_value)

    print('H0: (null hypothesis) Sample data comes from same distribution as theoretical distribution.')
    print('H1: (alternative hypothesis) Sample data does not comes from same distribution as theoretical distribution.')

    if p_value >= 0.05:
        print('we can not reject the null hypothesis')
    else:
        print('we reject the null hypothesis')

def chi2_gamma(sd,ss,nb):
    # observed
    print("Starting Gamma Fit... ")
    plt.figure()
    binEdges = np.linspace(0.0, np.max(sd), nb)
    observed, _ = np.histogram(sd, bins=binEdges)

    # MLE 
    fit_alpha, fit_loc, fit_beta = stats.gamma.fit(sd, floc=0)
    print(f"Gamma: Alpha: {fit_alpha} Beta: {fit_beta}")

    # expected
    expectedProb = stats.gamma.cdf(binEdges, fit_alpha, scale=fit_beta, loc=fit_loc)
    expectedProb[-1] += 1.0 - np.sum(np.diff(expectedProb))
    expected = ss * np.diff(expectedProb)

    binMidPt = (binEdges[1:] + binEdges[:-1]) / 2
    plt.hist(sd, bins=binEdges, label='Observed')
    plt.plot(binMidPt, expected, 'or-', label='Expected')
    plt.plot(binMidPt, observed, 'oy-', label='Observed')
    plt.legend()
    plt.title("Gamma")

    chiSq, pValue = stats.chisquare(f_obs=observed, f_exp=expected)
    print(f'ChiSquare Statistic {chiSq:0.3f} P value {pValue:0.3f}')

    print('H0: (null hypothesis) Sample data follows the hypothesized distribution.')
    print('H1: (alternative hypothesis) Sample data does not follow a hypothesized distribution.')

    if pValue >= 0.05:
        print('we can not reject the null hypothesis')
    else:
        print('we reject the null hypothesis')

def ks_gamma(sd, ss): 
    print("Starting Gamma KS Fit...")
    plt.figure()

    fit_alpha, fit_loc, fit_beta=stats.gamma.fit(sd, floc=0)
    print(f"Gamma: Alpha: {fit_alpha} Beta: {fit_beta}")

    x = np.linspace(stats.gamma.ppf(0.01, fit_alpha),stats.gamma.ppf(0.99, fit_alpha), ss)

    KS_stat, p_value = stats.kstest(sd, stats.gamma.cdf(x, a=fit_alpha))
    # arbitrary cdf tested
    sortedData = np.sort(sd)

    count = np.ones(ss)
    count = np.cumsum(count) / ss

    cdf = stats.gamma.cdf(sortedData, a=fit_alpha, loc=fit_loc, scale=fit_beta)

    plt.plot(sortedData, count, 'b', label='sample data')
    plt.plot(sortedData, cdf, 'r', label='theoretical data')
    plt.legend()
    plt.xlabel('data sample')
    plt.ylabel('cummulative frequency')
    plt.title('Gamma KS')

    KS_stat, p_value = stats.ks_2samp(count, cdf)

    print('KS statistic', KS_stat)
    print('p value', p_value)

    print('H0: (null hypothesis) Sample data comes from same distribution as theoretical distribution.')
    print('H1: (alternative hypothesis) Sample data does not comes from same distribution as theoretical distribution.')

    if p_value >= 0.05:
        print('we can not reject the null hypothesis')
    else:
        print('we reject the null hypothesis')

num = 1
fileName = ""
sampleData, sampleSize = process_data("data", num)
if num == 0:
    fileName = "SelfCheckout"
elif num == 1:
    fileName = "CashierCheckout"
Output_Path_Parent = "./FitTests"
Output_Path = f"./{Output_Path_Parent}/{fileName}-Data"

if not (os.path.exists(Output_Path_Parent)):
    os.mkdir(Output_Path_Parent)

if not (os.path.exists(Output_Path)):
    os.mkdir(Output_Path)

bins = [20] # np.linspace(1,sampleSize).astype(int)
if num == 0:
    bins = [35]
    
# with open(f'{Output_Path}/FitTests_{fileName}.txt', 'w') as sys.stdout:

print(f"Starting Fit Tests for -{fileName}- data \n")

for bin in bins:

    #################################
    # Exponential Chi Squared Testing 
    #################################

    # fit = "Exponential"
    # print(f"Number of bins: {bin}\n")
    # chi2_expon(sampleData,sampleSize,bin)
    print()

    # plt.xlabel("Data Value")
    # plt.ylabel("Data Frequency")
    # plt.title(f"{fit} Fit for {fileName} Data")
    # plt.savefig(f"{Output_Path}/{fileName}_Chi_{bin}-Bins")
    
    #############################
    # Uniform Chi Squared Testing 
    #############################
    # chi2_uniform(sampleData,sampleSize,bin)
    # ks_uniform(sampleData, sampleSize)
    # print()
    
    #############################
    # Weibull Chi Squared Testing 
    #############################
    # chi2_weibull(sampleData,sampleSize,bin)
    # ks_weibull(sampleData, sampleSize)
    # print()
    
    #################################
    # Lognormal Chi Squared Testing 
    #################################
    # plt.figure()
    # fit = "Lognormal"
    # print(f"Number of bins: {bin}\n")
    # chi2_lognorm(sampleData,sampleSize,bin)
    # print()

    # plt.xlabel("Data Value")
    # plt.ylabel("Data Frequency")
    # plt.title(f"{fit} Fit for {fileName} Data")
    # plt.savefig(f"{Output_Path}/{fileName}_Chi_{bin}-Bins")

    #################################
    # Gamma Chi Squared Testing 
    #################################
    
    # chi2_gamma(sampleData,sampleSize,bin)
    # ks_gamma(sampleData,sampleSize)
    # print()
    
#################################
# Exponential KS Fit Testing 
#################################

# fit = "Exponential KS"
# ks_expon(sampleData, sampleSize)
# plt.xlabel("Data Value")
# plt.ylabel("Cumulative Frequency")
# plt.title(f"{fit} Fit for {fileName} Data")
# plt.savefig(f"{Output_Path}/{fileName}_KS")

#################################
# Lognormal KS Fit Testing 
#################################

# fit = "Lognormal KS"
# ks_lognorm(sampleData, sampleSize)
# plt.xlabel("Data Value")
# plt.ylabel("Cumulative Frequency")
# plt.title(f"{fit} Fit for {fileName} Data")
# plt.savefig(f"{Output_Path}/{fileName}_KS")
        
# plt.show()

chiSq, pval = stats.chisquare([115, 96, 110, 89, 91, 110, 98, 97, 103, 91])
print(f'ChiSquare Statistic {chiSq}, P Value {pval}')