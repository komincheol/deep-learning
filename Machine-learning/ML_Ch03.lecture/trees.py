'''
Created on Oct 12, 2010
Decision Tree Source Code for Machine Learning in Action Ch. 3
@author: Peter Harrington
'''
from math import log
import operator

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    #change to discrete values
    return dataSet, labels

def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet: #the the number of unique elements and their occurance
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2) #log base 2
    return shannonEnt


def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]     #chop out axis used for splitting
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def chooseBestFeatureToSplit_lec(dataSet):
    numFeatures = len(dataSet[0]) - 1      #the last column is used for the labels
    baseEntropy = calcShannonEnt(dataSet)
    print "== Begin finding the best feature =="
    print "Feature #: {}, Base Entropy: {}".format(numFeatures, baseEntropy)
    bestInfoGain = 0.0;
    bestFeature = -1
    
    for i in range(numFeatures):        #iterate over all the features
        featList = [example[i] for example in dataSet]#create a list of all the examples of this feature
        uniqueVals = set(featList)       #get a set of unique values
        newEntropy = 0.0
        print "\n  feature {}'s value list: {}".format(i, featList)
        print "  Unique value of the above feature: {}".format(uniqueVals)
        
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            print "\n    SubDataSet(feat#: {}, feat-val: {}): {}".format(
                i, value, subDataSet)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
            print "    Probability of the subdataset: {}".format(prob)
            print "    Entropy of the subdataset: {}".format(calcShannonEnt(subDataSet))
            print "    Cumulative Entropy: {}".format(newEntropy)
                
        infoGain = baseEntropy - newEntropy     #calculate the info gain; ie reduction in entropy
        print "\nThe determined Entropy for feature {}: {}".format(i, newEntropy)
        print "Information Gain for feature {}: {}".format(i, infoGain)
                                                             
        if (infoGain > bestInfoGain):       #compare this to the best gain so far
            bestInfoGain = infoGain         #if better than current best, set to best
            bestFeature = i
            
    print "\nThe best feature returned: '{}'".format(bestFeature)
    return bestFeature                      #returns an integer

def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1      #the last column is used for the labels
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0;
    bestFeature = -1
    
    for i in range(numFeatures):        #iterate over all the features
        featList = [example[i] for example in dataSet]#create a list of all the examples of this feature
        uniqueVals = set(featList)       #get a set of unique values
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)     
        infoGain = baseEntropy - newEntropy     #calculate the info gain; ie reduction in entropy
        if (infoGain > bestInfoGain):       #compare this to the best gain so far
            bestInfoGain = infoGain         #if better than current best, set to best
            bestFeature = i
    return bestFeature                      #returns an integer


def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createTree_lec(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    print "=== Begin createTree() ==="
    print "Class labels for training set: {}".format(classList)
    
    # Recursive Stop Condition #1
    # stop splitting when all of the classes are equal
    if classList.count(classList[0]) == len(classList): 
        print "\r  StopCondition #1: Leaf Node's value: {}".format(classList[0])
        return classList[0]
    
    # Recursive Stop Condition #1
    # stop splitting when all of the classes are equal
    if len(dataSet[0]) == 1: #stop splitting when there are no more features in dataSet
        print "\r  StopCondition #2: Leaf Node's value: {}".format(majorityCnt(classList))
        return majorityCnt(classList)
    
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    print "\r  Current Best Feature: {}".format(bestFeatLabel)
    
    myTree = {bestFeatLabel:{}}
    print "    Current Tree: {}".format(myTree)
    
    del(labels[bestFeat])
    print "    Class labels remained: {}".format(labels)
    
    featValues = [example[bestFeat] for example in dataSet]
    print "    \rDataSet for best feature: {}".format(featValues)
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]       #copy all of labels, so trees don't mess up existing labels
        print "      \rsublabels: {}".format(subLabels)
        myTree[bestFeatLabel][value] = createTree_lec(splitDataSet(dataSet, bestFeat, value),subLabels)
    return myTree                            
    
def classify_lec(inputTree,featLabels,testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    print "firstStr: {}".format(firstStr)
    print "secondDict: {}".format(secondDict)
    print "featIndex: {}".format(featIndex)
    print "key: {}".format(testVec[featIndex])
    print "valueOfFeat: {}".format(secondDict[key])
    
    if isinstance(valueOfFeat, dict): 
        classLabel = classify_lec(valueOfFeat, featLabels, testVec)
    else:
        classLabel = valueOfFeat
    
    return classLabel

def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()
    
def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)


def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList): 
        return classList[0]#stop splitting when all of the classes are equal
    if len(dataSet[0]) == 1: #stop splitting when there are no more features in dataSet
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]       #copy all of labels, so trees don't mess up existing labels
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels)
    return myTree                            

def classify(inputTree,featLabels,testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    
    if isinstance(valueOfFeat, dict): 
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:
        classLabel = valueOfFeat
    
    return classLabel

