# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 11:24:46 2019

@author: kyrie
"""

with open("CreditCards.csv") as file:
    content = []
    x = []
    each = [[0 for i in range(690)]for j in range(14)]
    
    for i in file:
        i = i.strip().split(',')
        x.append(i)
    x = x[1:]
    print(len(x))
    a = 0
    for i in range(len(x)):
        #i=0
        
        
        for j in range(14):
            each[j][i] = x[i][a]
 #      for j in x[i]: j 01 2 3 4 5
 #          each.append(j)
        if a<14:
            a+=1
    print(each)