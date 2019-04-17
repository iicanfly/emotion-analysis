# coding: utf-8




import nltk
#nltk.download('stopwords')
import io
import time
import math
import random
from numpy import *
from nltk.corpus import stopwords
# from NN import Network
from NeuralNetwork import Network
import sys
reload(sys) 
sys.setdefaultencoding('utf-8')




# 初步处理原始数据
def deal_data(data):
    remove_set = ['<br />', '.', ',', '!', '?', '/', ';', '\'', '*', '"', '(', ')', '-', '，']
    #英文停止词，set()集合函数消除重复项
    remove_word = list(set(stopwords.words('english')))
    #data所有的符号使用空格代替
    for each in remove_set:
        data = data.replace(each, ' ')
    #data以\n(段)分隔开
    data = data.split('\n')
    #对每一段进行循环
    for i in range(len(data)):
        #每一段文本以空格进行分割
        data[i] = data[i].split(' ')
        remove_list = []
        #对一段的每一个字符循环
        for j in range(len(data[i])):
            if (data[i][j].isalpha() or data[i][j].isalnum()) and (len(data[i][j]) > 1):
                #大写变小写
                data[i][j] = data[i][j].lower()
            else:
                #若不是数字也不是字母，就加入remove_list当中
                if data[i][j] not in remove_list:
                    remove_list.append(data[i][j])

        #把文本当中的停止词全部去除
        data[i] = [x for x in data[i] if x not in remove_word]

        #若文本包含remove_list中的字符，则去除
        for each in remove_list:
            while each in data[i]:
                data[i].remove(each)

    return data




# 生成初步处理文件
def handleOrgData(ifileName, ofileName):
    start = time.clock()
    ifstream = io.open(ifileName, 'r', encoding = 'UTF-8')
    #读取文件到cont
    cont = ifstream.read()
    ifstream.close()
    #传入deal_data函数初步处理文本
    comment = deal_data(cont)
    ofstream = io.open(ofileName, 'w', encoding = 'UTF-8')
    #循环每一段文本
    for i in range(len(comment)):
        string = ''
        for each in comment[i]:
            string += str(each) + ' '
        if i < len(comment) - 1:
            string += '\n'
        ofstream.write(string.decode('utf-8'))
    ofstream.close()
    print(time.clock() - start)
    return




# 读取处理好的文件
def readTreatedData(ifileName):
    ifstream = io.open(ifileName, 'r', encoding = 'UTF-8')
    data = []
    #以行为单位进行存储
    for line in ifstream:
        data.append(line.split())
    ifstream.close()
    return data




# 读入标签
def readLabel(ifileName):
    ifstream = io.open(ifileName, 'r', encoding = 'UTF-8')
    label = []
    for line in ifstream:
        subList = []
        line = line.strip('\n')
        subList.append(int(line))
        label.append(subList)
    ifstream.close()
    return label




# 分割训练集，测试集
def splitDataSet(orgData, label, splitRate, train=True):
    orgCombineData = list(zip(orgData, label))
    if train == 1:
        random.shuffle(orgCombineData)
    splitIndex = int(len(orgData) * (splitRate * 1.0 / (splitRate + 1)))
    print "splitIndex=",splitIndex
    return orgCombineData[:splitIndex], orgCombineData[splitIndex + 1:]




# 建立词库
def buildWordLib(fileName):
    wordSet = dict()
    ifstream = io.open(fileName, 'r', encoding = 'UTF-8')
    for line in ifstream:
        line = line.strip('\n')
        line = line.split()
        for word in line:
            wordSet[word] = 0
    ifstream.close()
    return wordSet




# 测试字典序
def testOutput(filename, dict):
    ofstream = io.open(filename, 'w', encoding = 'UTF-8')
    for key, value in dict.items():
        ofstream.write((key.decode('utf-8') + u':' + str(value).decode('utf-8')))
        ofstream.write(u'\n')
    ofstream.close()




# 建立测试集的one-hot矩阵
def buildOneHotTest(orgCombineData, wordLib):
    testData = []
    # count = 0
    #x为文本,y为label标签
    for x, y in orgCombineData:
        # count = count + 1
        libTmp = wordLib.copy()
        for word in x:
            if word in libTmp:
                libTmp[word] = 1
        # if (count < 10):
        # 	testOutput("testDict%d.txt" % (count + 1), libTmp)
        testData.append((mat(list(libTmp.values())).transpose(), mat(y)))
    return testData




#建立预测集的one-hot矩阵
def predictData(preData, wordLib):
    oneHot = []
    for data in preData:
        libTmp = wordLib.copy()
        for word in data:
            if word in libTmp:
                libTmp[word] = 1
        oneHot.append(mat(list(libTmp.values())).transpose())
    return oneHot




# 建立one-hot矩阵的同时训练数据
# 由于one-hot矩阵过大，因此最好得到一个就去训练一次
def trainStep(train, test, preData, wordLib, net):
    #testOutput("dictOrg.txt", wordLib)
    print("begin train")
    count = 0
    pre = 0
    accuracyValue = [0, 0]
    combList = []
    for trainData, trainLabel in train:
# 		print("this")
        count = count + 1
        # if(count>9):
        # 	return
        libTmp = wordLib.copy()
        for word in trainData:
            if (word in libTmp):
                libTmp[word] = 1
        # testOutput("trainDict%d.txt" % (count + 1), libTmp)
        combData = (mat(list(libTmp.values())).transpose(), mat(trainLabel))
        combList.append(combData)
        if count == len(train):
            print("第%d-%d个训练样本: " % (pre, count))
            pre = count
            start = time.clock()
            accuracyValue = net.SGD(combList, 40, 128, 0.01, accuracyValue, predictData = preData,
                                    evaluation_data = test,
                                    monitor_evaluation_cost = True,
                                    monitor_evaluation_accuracy = True,
                                    monitor_training_cost = True,
                                    monitor_training_accuracy = True)
            print("消耗时间： " + str(time.clock() - start))
            combList = []
    
#     print combList[0]
    print("end train")
    return




if __name__ == '__main__':
# 	handleOrgData("./2/trainData.txt","./2/train_data.txt")
# 	handleOrgData("./2/testData.txt","./2/test_data.txt")

    # 读入训练集
    orgData = readTreatedData("./2/train_data.txt")
    # 读入预测集
    preData = readTreatedData("./2/test_data.txt")
    # 读入标签集
    label = readLabel("./2/trainLabel.txt")

    print "orgData_type=",type(orgData)
    print "len(orgData)=",len(orgData)
    print "preData_type=",type(preData)
    print "len(preData)=",len(preData)
    print "label_type=",type(label)
    print "len(label)=",len(label)

    #划分训练集，测试集
    trainData, testData = splitDataSet(orgData, label, 47)

    print "trainData=",type(trainData)
    print "len(trainData)=",len(trainData)
    print 'len(trainData[0])=',len(trainData[0])
    print 'len(trainData[0][0])=',len(trainData[0][0])
    print 'len(trainData[0][1])=',len(trainData[0][1])
    print "testData=",type(testData)
    print "len(testData)=",len(testData)

    wordLib = buildWordLib("./2/train_data_last.txt")

    print 'len(wordLib)=',len(wordLib)

    # 构建预测集的one-hot
    preOneHot = predictData(preData, wordLib)

    print 'type(preOneHot)=',type(preOneHot)
    print 'len(preOneHot)=',len(preOneHot)
    print 'len(preOneHot[0])=',len(preOneHot[0])
    print 'len(preOneHot[1])=',len(preOneHot[1])

    oneHotTest = buildOneHotTest(testData, wordLib)

    print 'type(oneHotTest)=',type(oneHotTest)
    print 'len(oneHotTest)=',len(oneHotTest)
    print 'len(oneHotTest[0])=',len(oneHotTest[0])
    print 'len(oneHotTest[0][0])=',len(oneHotTest[0][0])
    print 'len(oneHotTest[0][1])=',len(oneHotTest[0][1])

    wordSetLen, labelNum = len(wordLib), len(label[0])

    print 'wordSetLen=',wordSetLen
    print 'labelNum=',labelNum

    net = Network([wordSetLen, 64, 64, 64, labelNum])

    trainStep(trainData, oneHotTest, preOneHot, wordLib, net)

