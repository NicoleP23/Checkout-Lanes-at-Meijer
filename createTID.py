"""
@author: Nicole Pililyan 
        (and Ryan Miller, code taken and tweaked from Project 2)

This file creates a theoretical input distribution reflecting project distributions
 for the two collected data sets (self-checkout and cashier checkout)
"""

import numpy as np
import matplotlib.pyplot as plt
import random

# Method for creating a Theoretical Input Distribution reflecting project data

# createTID - Create Theoretical Input Distribution
# dt - The type of data being processed: Self-checkout = 0, Cashier checkout = 1
# num - Number of datapoints to generate
# alt - alternative, set to 1 to enable extended information, set 2 to create plots 
def createTID(dt, num, alt=0):
    times = []
    type = ""

    # Generate a lognormal variate representing a number of seconds
    if dt == 0:
        type = "Self-Checkout"
        for i in range (num):
            times.append(random.lognormvariate(np.log(179.9979),0.3614))
    elif dt == 1:
        type = "Cashier-Checkout"
        for i in range (num):
            times.append(random.lognormvariate(np.log(168.2403),0.3816))
    
    mean = np.mean(times)

    for i,j in enumerate(times):
        if j > 300:
            times[i] = round(mean)

    if alt:
        sampleMean = np.mean(times)
        sampleSTD = np.std(times)
        print(f'Theoretical Distribution Generated to Represent the {type} data:')
        print(f'sample mean: {sampleMean:0.3f}')
        print(f'sample std: {np.sqrt(sampleSTD):0.3f}\n')

    if alt == 2:
        # plot the raw data
        plt.figure()
        plt.bar(range(len(times)), times)
        plt.xlabel("Data Index")
        plt.ylabel("Data Value")
        plt.title(f"Generated Raw {type} Data via Theoretical Distribution")

        # plot a histogram

        plt.figure()
        numBins = 50
        plt.hist(times, numBins, density=True, align='mid')
        plt.xlabel("Data Value")
        plt.ylabel("Probability Density")
        plt.title(f"Binned Generated {type} Data via Theoretical Distribution")

    sampleData = times
    sampleSize = len(times)


    return sampleData, sampleSize

dist = 0
mean, std = createTID(dist,300, 1)

# if dist == 0:
#     print(f"Self-Checkout:\nMean: {mean}\nStd: {std}")
# elif dist == 1:
#     print(f"Cashier Checkout:\nMean: {mean}\nStd: {std}")
plt.show()
