# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 21:28:45 2019

@author: kyrie
"""

import numpy as np
import matplotlib.pyplot as plt
import math

with open('Advertising.csv','r') as file:
    tv = []
    radio = []
    newspaper = []
    sales =[]
    splited = []
    for row in file:
        row = row.strip('\n')
        splited = row.split(',')#[tv,ratio,]
        if splited[1] != 'TV':
            tv.append(float(splited[1]))
        if splited[2] != 'Radio':
            radio.append(float(splited[2]))
        if splited[3] != 'Newspaper':
            newspaper.append(float(splited[3]))
        if splited[4] != 'Sales':
            sales.append(float(splited[4]))
    
    
def zero_one_format(tempp):
    changed = []
    for i in tempp:
        num = (i-min(tempp))/(max(tempp)-min(tempp))
        changed.append(round(num,2)) 
    return changed
        
newspaper = zero_one_format(newspaper)
newspaper_training = newspaper[:190]
newspaper_testing = newspaper[190:]

print(sales)

theta_0 = -1
theta_1 = -0.5
learning_rate = 0.01
loss = []
for i in range(500):
    error = []  
    error_1 = []
    for j in range(190):
        predicted = theta_0 +theta_1 * newspaper_training[j]
        error.append(sales[j] - predicted)
        error_1.append((sales[j] - predicted)*newspaper_training[j])
    theta_0 = theta_0 + learning_rate * (np.mean(error))
    theta_1 = theta_1 + learning_rate * (np.mean(error_1))
    loss.append([np.mean(error)**2])


###print(theta_1)
plt.plot(loss)
plt.show()
print(f'The third time:\ntheta_0: {theta_0}')
print(f'theta_1: {theta_1}')

###rmse
###for test set
error_rmse = []
for i in range(len(newspaper_training)):
    predicted = theta_0 + theta_1 * newspaper_training[i]
    temp = (sales[i] - predicted)**2
    error_rmse.append(temp)

print(f'newspaper training set rmse: {math.sqrt(np.mean(error_rmse))}')
error_rmse = []
for i in range(len(newspaper_testing)):
    predicted = theta_0 + theta_1 * newspaper_testing[i]
    temp = (sales[190+i] - predicted)**2
    error_rmse.append(temp) 
print(f'newspaper testing set rmse: {math.sqrt(np.mean(error_rmse))}')


