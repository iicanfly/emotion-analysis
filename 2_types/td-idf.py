import numpy as np
import random
import sys
import io
import json
import math
reload(sys) 
sys.setdefaultencoding('utf-8')

def buildWordLib(fileName):
    wordSet = dict()
    ifstream = io.open(fileName, 'r', encoding = 'UTF-8')
    for line in ifstream:
        line = line.strip('\n')
        line = line.split()
        for word in line:
            wordSet[word] = np.array([0,0])
    ifstream.close()
    return wordSet

def tf_idf(infile, outfile,num):
    wordSet = buildWordLib(infile)
#     for x in wordSet:
#         print type(x)
    ifstream = io.open(infile,'r',encoding = 'utf-8')
    cont = ifstream.read()
    ifstream.read()
    cont = cont.split('\n')
    data = {}
    for i in range(len(cont)):
        cont[i] = cont[i].split(' ')
#     print wordSet
#     print len(cont)
#     print cont 
    count_a = 0
    count_b = 0
    count_c = 0
#     for x in wordSet:
#         count_a = count_a + 1
#         if count_a % 1000 == 0:
#             print count_a
#         for i in range(len(cont)):
#             if x in cont[i]:
#                 wordSet[x] = wordSet[x] + np.array([1,0])
#             for j in range(len(cont[i])):
#                 if x == cont[i][j]:
#                     wordSet[x] = wordSet[x] + np.array([0,1])

    for i in range(len(cont)):
        count_a = count_a + 1
        if count_a % 1000 == 0:
            print "count_a",count_a
        flag = dict()
#         print cont[0]
        for x in cont[i]:
#             print (x)
            if x == u'':
                break
            if x in flag:
                wordSet[x] = wordSet[x] + np.array([0,1])
            else:
                flag[x] = 1
                wordSet[x] = wordSet[x] + np.array([0,1])
                wordSet[x] = wordSet[x] + np.array([1,0])
            
#     print wordSet
    for x in wordSet:
        data[x] = 0.0
    total = len(cont)
    for x in wordSet:
        data[x] = float(wordSet[x][1]) * (math.log((total + 1.0)/(float(wordSet[x][0]) + 1.0)) + 1.0)
    data_last = {}
    for i in range(num):
        count_b = count_b + 1
        if count_b % 1000 == 0:
            print "b",count_b
        max_data = 0
        max_word = u''
        for x in data :
            if x not in data_last and data[x] > max_data:
                max_data = data[x]
                max_word = x
        data_last[max_word] = max_data
        
    ofstream = io.open(outfile,'w',encoding = 'utf-8')
    for i in range(len(cont)):
        count_c = count_c + 1
        if count_c % 1000 == 0:
            print "c",count_c
        string = ''
        for each in cont[i]:
            if each in data_last:
                string += each + ' '
        if i < len(cont) - 1:
            string += '\n'
        ofstream.write(string.decode('utf-8'))
    ofstream.close()
        
        
    
#     for x in data_last:
#         print x,data_last[x] 
#     for x in data:
#         print x, data[x]
#     print data
        
tf_idf('./2/train_data.txt','./2/train_data_last.txt',23151)
# tf_idf('./text_test.txt','./text_test_last',23)