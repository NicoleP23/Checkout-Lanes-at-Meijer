""" 
@author: Nicole Pililyan 

Simulation Model Experiment

Notes: 
    - CO stands for Checkout
    - Every simulation time unit is like a second in real time units
""" 

# imports
import simpy
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

class Customer:
    
    def __init__(self, number, env, cashier, register): 

        self.custNo = number 
        self.env = env 
        self.cashier = cashier
        self.register = register
        self.timeline = []


    def shortest_line(self):
        cashierNo = 0
        
        if cashiers[0].count < cashiers[0].capacity: 
            cashierNo = 0 
        elif cashiers[1].count < cashiers[1].capacity: 
           cashierNo = 1 
        elif cashiers[2].count < cashiers[2].capacity: 
           cashierNo = 2
        elif cashiers[3].count < cashiers[3].capacity: 
           cashierNo = 3
        elif len(cashiers[0].queue) < len(cashiers[1].queue) and len(cashiers[0].queue) < len(cashiers[2].queue) and len(cashiers[0].queue) < len(cashiers[3].queue): 
            cashierNo = 0 
        elif len(cashiers[1].queue) < len(cashiers[2].queue) and len(cashiers[1].queue) < len(cashiers[3].queue): 
            cashierNo = 1
        elif len(cashiers[2].queue) < len(cashiers[3].queue): 
            cashierNo = 2
        else: 
            cashierNo = 3
       
        return cashierNo # return result
    
    def self_checkout(self):
        # mPerks
        options = [0, 1] # 0 means no mPerks, 1 means the customer used mPerks
        weights = [49, 51] # 49% of the time won't have mPerks, 51% of the time will
        mPerks = random.choices(options, weights)
        # print(f"mPerks: {mPerks}")
        
        # needing assistance
        options = [0, 1] # 0 means no assistance, 1 means the customer needed employee help
        weights = [60, 40] # 40% of the time will need assistance
        assist = random.choices(options, weights)
        # print(f"Assist: {assist}")
        
        # get in line for the self-checkout machines
        req = register.request()
        yield req
        
        # start checking out    
        startCOTime = self.env.now    
        self.timeline.append(f"Customer{self.custNo} goes to a register and begins self-checkout at {startCOTime:7.3f}")
        # print(f'customer{self.custNo} starts checkout with cashier {cashierNo} at {startCOTime:7.3f}') 
        timeTaken = random.lognormvariate(np.log(179.99799649904895),0.3614074573845922)        
        yield self.env.timeout(timeTaken)
        # print(f"Checkout Time: {timeTaken}")
        self.timeline.append(f"{self.custNo} Checkout Time: {timeTaken}")
        register.release(req) 
        
        endCOTime = self.env.now
        totalCOTime = endCOTime - startCOTime
        
        # after getting the amount of time it took the customer to checkout,
        # add time to how long it took to checkout based on variables
        if mPerks[0] == 1:
            # print(f"{self.custNo} had mPerks")
            self.timeline.append(f"{self.custNo} had mPerks")
            # print(f"{self.custNo} had mPerks, total time before: {totalCOTime}")
            totalCOTime += 15
            # print(f"{self.custNo} total time after: {totalCOTime}")
        
        if assist[0] == 1:
            # print(f"{self.custNo} needed assistance")
            self.timeline.append(f"{self.custNo} needed assistance")
            # add ## seconds to total checkout time
            waitForAssist =random.expovariate(1/21.2) # waiting for assistance
            if waitForAssist < 5 or waitForAssist > 49:
                waitForAssist = 21.2 # change to mean amount of time if the value is too small or too big
            totalCOTime += waitForAssist
            # totalCOTime += 21.16
            # getAssist = random.expovariate(1/20.5) # getting assistance
            # totalCOTime += getAssist
            totalCOTime += 20.5
            # self.timeline.append(f"{self.custNo} waited for {waitForAssist}s and got help for {getAssist}s long")
            # self.timeline.append(f"{self.custNo} waited for {waitForAssist}s")
            
        self.timeline.append(f"Customer{self.custNo} finishes checkout at {endCOTime:7.3f} and exits")
        
        # uncomment these lines to read the timeline
        # for string in self.timeline:
        #     print(string)
        
        COTimeList.append(totalCOTime)
            
    
    def cashier_checkout(self):
        # mPerks
        options = [0, 1] # 0 means no mPerks, 1 means the customer used mPerks
        weights = [57, 43] # 57% of the time won't have mPerks, 43% of the time will
        mPerks = random.choices(options, weights)
        
        # pick the cashier with the shortest line
        cashierNo = self.shortest_line()
        req = cashiers[cashierNo].request() 
        yield req
    
        # start checking out    
        startCOTime = self.env.now    
        self.timeline.append(f"Customer{self.custNo} starts checkout with cashier {cashierNo} at {startCOTime:7.3f}")
        # print(f'customer{self.custNo} starts checkout with cashier {cashierNo} at {startCOTime:7.3f}') 
        timeTaken = random.lognormvariate(np.log(168.24036960493748),0.3816123378187324)
        yield self.env.timeout(timeTaken)
        # print(f"Checkout Time: {timeTaken}")
        self.timeline.append(f"Checkout Time: {timeTaken}")
        cashiers[cashierNo].release(req) 
        
        endCOTime = self.env.now
        totalCOTime = endCOTime - startCOTime
        if mPerks[0] == 1:
            self.timeline.append(f"{self.custNo} had mPerks")
            # print(f"{self.custNo} had mPerks, total time before: {totalCOTime}")
            totalCOTime += 15
            # print(f"{self.custNo} total time after: {totalCOTime}")
            
        self.timeline.append(f"Customer{self.custNo} finishes checkout with cashier {cashierNo} at {endCOTime:7.3f} and exits")
        # print(f"customer{self.custNo} finishes checkout with cashier {cashierNo} at {endCOTime:7.3f} and exits") 
        # print(f"The time it took {self.custNo} to checkout was: {totalCOTime}")
        
        # uncomment these lines to read the timeline
        # for string in self.timeline:
        #     print(string)
        
        COTimeList.append(totalCOTime)
        
    def setTotalCust(amount):
        global totalCustNo
        if amount > totalCustNo or amount == 1:
            totalCustNo = amount
        # print(f"The amount {amount}")
        # print(totalCustNo)
        
        
    def getTotalCust():
        return totalCustNo
    
def customerGenerator(env, cashier, register): 
    custNo = 0 
    global totalCustNo
    while True: 
        cust = Customer(custNo, env, cashier, register)
        # env.process(cust.self_checkout()) # self-checkout process
        env.process(cust.cashier_checkout()) # cashier checkout process
        custNo += 1 
        yield env.timeout(random.expovariate(1/180.0)) # customer arrives on an average (every 3 min (180s))
        totalCustNo = 0
        Customer.setTotalCust(custNo)
    
    
everyCOTime = []
averageCOTime = [] 
avgAmountCust = []
avgSEM = []
for replicate in range(300): 
    COTimeList = [] 
    env = simpy.Environment() 
    
    # there are 4 cashiers available to checkout at
    cashiers = [simpy.Resource(env, capacity=1), simpy.Resource(env, capacity=1), simpy.Resource(env, capacity=1), simpy.Resource(env, capacity=1)] 
    # 1 line for 6 check-out machines
    register = simpy.Resource(env, capacity=6) 
    env.process(customerGenerator(env, cashiers, register)) 
    env.run(until=7200.0) 
    #print(COTimeList) 
    # print(f'average checkout time is {np.average(COTimeList)}') 
    everyCOTime.append(COTimeList)
    averageCOTime.append(np.average(COTimeList)) 
    avgSEM.append(stats.sem(COTimeList))
    
    totalCustomers = Customer.getTotalCust()
    # print(f'Amount of Customers is {totalCustomers}') 
    avgAmountCust.append(totalCustomers)
    
    
# type = "Self-Checkout"
type = "Cashier Checkout"
print(f"Average Overall {type} Checkout Time: {np.average(averageCOTime)}")
print(f"Average Overall Max {type} Checkout Time: {np.max(averageCOTime)}")
print(f"Average Overall Min {type} Checkout Time: {np.min(averageCOTime)}")
print(f"Average Overall SEM of {type} Checkout Time: {np.average(avgSEM)}")
# print(f"Average Amount of Customers per {type} Sim: {np.average(avgAmountCust)}")
print(f"Total Amount of Customers per {type} Sim: {np.sum(avgAmountCust)}")


plot_data = True
if plot_data:
    plt.figure() 
    plt.bar(range(len(averageCOTime)), averageCOTime)
    plt.ylabel('Average Checkout Time') 
    plt.xlabel('Experiment Number') 
    plt.title(f'Experiment Averages - {type}')
    
    plt.figure() 
    plt.hist(averageCOTime, bins=20) 
    plt.xlabel('Average Checkout Time') 
    plt.ylabel('Frequency') 
    plt.title(f'Experimental Results - {type}') 