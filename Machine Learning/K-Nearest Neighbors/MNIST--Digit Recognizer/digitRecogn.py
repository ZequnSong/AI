import csv
import operator
import time
import numpy as np

def loadData():
    trainList = []
    with open('data/train.csv') as datafile:
        lines = csv.reader(datafile)
        for line in lines:
            trainList.append(line)
    trainList.remove(trainList[0])
    trainList = np.mat(np.array(trainList))
    m,n = np.shape(trainList)
    data = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            if j==0:
                data[i,j] = int(trainList[i,j])
            elif int(trainList[i,j]) != 0:
                data[i,j] = 1
    return data

# testData : one row for a single test case's pixel data
# trainDataSet : entire train dataset we are using
# trianLabel : Lable for the train dataset
def knn(testData, trainDataSet, trainLabel):

    dataSetSize = trainDataSet.shape[0] # number of training data
    diffMat = np.tile(testData, (dataSetSize,1)) - trainDataSet # differences matrix between current test data and each training data 
    sqDiffMat = np.array(diffMat)**2 # square the differences matrix
    sqDistances = sqDiffMat.sum(axis=1) # sum the square differences
    distances = sqDistances**0.5 # square root to get the Euclidean Distance
    sortedDistIndex = distances.argsort() # sort the distances
    countLabel = {} # save the number of occurrence for each possible label
    for i in range(k):
        curLabel = trainLabel[sortedDistIndex[i]]
        # get count of corresponding label in dictionary, return 0 if not exist
        countLabel[curLabel] = countLabel.get(curLabel, 0) + 1
    rankedLabel = sorted(countLabel.items(), key=operator.itemgetter(1), reverse=True) 
    return int(rankedLabel[0][0])

# ------loading data----------
start1 = time.perf_counter()
trainData = loadData()
size = trainData.shape[0]
testData = trainData[int(size*0.8):,1:]
testLabel = trainData[int(size*0.8):,0]
trainLabel = trainData[0:int(size*0.8),0]
trainData = trainData[0:int(size*0.8),1:]
testSize = testData.shape[0]
k = 3
errorCount = 0
confusionMat = np.zeros((10,10))
start2 = time.perf_counter()
for i in range(testSize):
    print("Predicting # ",i+1," test data ...")
    curRes = knn(testData[i], trainData, trainLabel)
    trueRes = int(testLabel[i])
    if(curRes != trueRes):
        errorCount += 1
    confusionMat[curRes,trueRes] += 1
print("----Confusion Matrix----")
print(confusionMat)
print("accurancy is: %f" %((testSize-errorCount)/float(testSize)))
print("runtime is ",time.perf_counter()-start1)
print("actual wall time is ",time.perf_counter()-start2)
