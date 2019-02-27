from math import log
import operator
import random
import numpy
import math
import  matplotlib.pyplot as plt
from matplotlib.ticker import  MultipleLocator

# def generateDataSet(k, m):
#     data = [[0] * (k + 1) for i in range(m)]
#     labels = ['X' + str(i + 1) for i in range(k)]
#     for i in range(m):
#         for j in range(k):
#             if j == 0 :
#                 data[i][j] = 0 if random.uniform(0, 1) < 0.5 else 1
#             else :
#                 data[i][j] = data[i][j - 1] if random.uniform(0, 1) < 0.75 else  data[i][j - 1] ^ 1
#
#     # calculate Y
#     divider = sum([0.9 ** i for i in range(2, k + 1)])
#     w = [(0.9 ** i) / divider for i in range(2, k + 1)]
#     w.insert(0, 0)
#
#     for rowVector in data:
#         r = rowVector[:k + 1]
#         temp = sum(i[0] * i[1] for i in zip(r, w))
#         rowVector[-1] = rowVector[0] if temp >= 0.5 else 1 - rowVector[0]
#     return data, labels



def generateDataSet(m):
    k = 21
    data = [[0] * (k + 1) for i in range(m)]
    labels = ['X' + str(i) for i in range(k)]

    for i in range(m):
        for j in range(k):
                    if j == 0 :
                        data[i][j] = 0 if random.uniform(0, 1) < 0.5 else 1
                    elif j > 0 and j <= 14:
                        data[i][j] = data[i][j - 1] if random.uniform(0, 1) < 0.75 else (1 - data[i][j - 1])
                    else:
                        data[i][j] = 0 if random.uniform(0, 1) < 0.5 else 1
    for rowVector in data:
        if rowVector[0] == 0:
            X_1_to_7 = rowVector[1:8]
            rowVector[-1] = 0 if X_1_to_7.count(0) > 3 else 1
        else:
            X_8_to_14 = rowVector[8:15]
            rowVector[-1] = 1 if X_8_to_14.count(1) > 3 else 0
    return data, labels



def splitDataSet(dataSet, index, target):
    subDataSet = []
    for rowVector in dataSet:
        if rowVector[index] == target:
            newRowVector = rowVector[:index]
            newRowVector.extend(rowVector[index + 1:])
            subDataSet.append(newRowVector)

    return subDataSet


def maximumInformationGainIndex(dataSet):
    infoContent = calculateInfoContent(dataSet)
    index = 0
    maxInfoGain = 0
    for i in range(len(dataSet[0]) - 1):
        entropy = calculateInfoGain(dataSet, i)
        infoGain = infoContent - entropy
        if infoGain > maxInfoGain:
            maxInfoGain = infoGain
            index = i
    return index


def calculateInfoGain(dataSet, index):
    colVector = [row[index] for row in dataSet]

    #count how many colVector == 1
    p_1 = sum(colVector) / len(colVector)
    p_0 = 1 - p_1
    entropy = 0

    subDataSet_0 = splitDataSet(dataSet, index, 0)
    subDataSet_1 = splitDataSet(dataSet, index, 1)
    entropy = p_1 * calculateInfoContent(subDataSet_1) + p_0 * calculateInfoContent(subDataSet_0)
    return entropy

def calculateInfoContent(dataSet):
    unique_Y = {}
    Y = [rowVector[-1] for rowVector in dataSet]
    for y in Y:
        if y not in unique_Y.keys():
            unique_Y[y] = 0
        unique_Y[y] += 1
    entropy = 0
    for key in unique_Y:
        probability = float(unique_Y[key]) / len(dataSet)
        entropy -= probability * math.log2(probability)
    return entropy


def buildDecisionTree(dataSet, labelCopy):
    Y = [rowVector[-1] for rowVector in dataSet]
    label = labelCopy[:]
    # return the leave node
    # if Y can only be 1 or 0, return the value, no need to split data
    if Y.count(Y[0]) == len(Y):
        return Y[0]

    if len(dataSet[0]) == 1:
        countYEqualTo1 = sum(Y)
        ans = 1 if countYEqualTo1 >= (len(Y) / 2) else 0
        return ans

    maxIndexOfInfoGain = maximumInformationGainIndex(dataSet)
    maxIndex_Label = label[maxIndexOfInfoGain]
    dcTree = {maxIndex_Label: {}}
    del(label[maxIndexOfInfoGain])

    selectedCol = [rowVector[maxIndexOfInfoGain] for rowVector in dataSet]
    setOfselectedCol = set(selectedCol)
    for val in setOfselectedCol:
        copyOflable = label[:]
        subDataSet = splitDataSet(dataSet, maxIndexOfInfoGain, val)
        dcTree[maxIndex_Label][val] = buildDecisionTree(subDataSet, copyOflable)
    return dcTree

def prunningDecisionTree_byDepth(dataSet, labelCopy, depth, maxDepth):
    Y = [rowVector[-1] for rowVector in dataSet]
    label = labelCopy[:]
    # return the leave node
    # reaches the max depth, then return the leave node
    if depth == maxDepth:
        countYEqualTo1 = sum(Y)
        ans = 1 if countYEqualTo1 >= (len(Y) / 2) else 0
        return ans

    # if Y can only be 1 or 0, return the value, no need to split data
    if Y.count(Y[0]) == len(Y):
        return Y[0]

    if len(dataSet[0]) == 1:
        countYEqualTo1 = sum(Y)
        ans = 1 if countYEqualTo1 >= (len(Y) / 2) else 0
        return ans

    maxIndexOfInfoGain = maximumInformationGainIndex(dataSet)
    maxIndex_Label = label[maxIndexOfInfoGain]
    dcTree = {maxIndex_Label: {}}
    del (label[maxIndexOfInfoGain])

    selectedCol = [rowVector[maxIndexOfInfoGain] for rowVector in dataSet]
    setOfselectedCol = set(selectedCol)
    depth += 1
    for val in setOfselectedCol:
        copyOflable = label[:]
        subDataSet = splitDataSet(dataSet, maxIndexOfInfoGain, val)
        dcTree[maxIndex_Label][val] = prunningDecisionTree_byDepth(subDataSet, copyOflable, depth, maxDepth)
    return dcTree


def prunningDecisionTree_bySize(dataSet, labelCopy, maxSize):
    Y = [rowVector[-1] for rowVector in dataSet]
    size = len(dataSet[0])
    label = labelCopy[:]
    # return the leave node
    # if Y can only be 1 or 0, return the value, no need to split data
    if size == maxSize:
        countYEqualTo1 = sum(Y)
        ans = 1 if countYEqualTo1 >= (len(Y) / 2) else 0
        return ans

    if Y.count(Y[0]) == len(Y):
        return Y[0]

    if len(dataSet[0]) == 1:
        countYEqualTo1 = sum(Y)
        ans = 1 if countYEqualTo1 >= (len(Y) / 2) else 0
        return ans

    maxIndexOfInfoGain = maximumInformationGainIndex(dataSet)
    maxIndex_Label = label[maxIndexOfInfoGain]
    dcTree = {maxIndex_Label: {}}
    del (label[maxIndexOfInfoGain])

    selectedCol = [rowVector[maxIndexOfInfoGain] for rowVector in dataSet]
    setOfselectedCol = set(selectedCol)
    for val in setOfselectedCol:
        copyOflable = label[:]
        subDataSet = splitDataSet(dataSet, maxIndexOfInfoGain, val)
        dcTree[maxIndex_Label][val] = prunningDecisionTree_bySize(subDataSet, copyOflable, maxSize)
    return dcTree

#
# def prunningDecisionTree_byChaiSquare(dataSet, labelCopy, t):
#     Y = [rowVector[-1] for rowVector in dataSet]
#     label = labelCopy[:]
#     independ_index = independIndex_bychaiSquare(dataSet, t)
#     # return the leave node
#     # if Y can only be 1 or 0, return the value, no need to split data
#     if Y.count(Y[0]) == len(Y):
#         return Y[0]
#
#     if len(dataSet[0]) == 1:
#         countYEqualTo1 = sum(Y)
#         ans = 1 if countYEqualTo1 >= (len(Y) / 2) else 0
#         return ans
#
#     if independ_index == -1:
#         # print('terminate at ' + label[0])
#         countYEqualTo1 = sum(Y)
#         ans = 1 if countYEqualTo1 >= (len(Y) / 2) else 0
#         return ans
#
#     # independIndex = independIndex_bychaiSquare(dataSet)
#     maxIndex_Label = label[independ_index]
#     dcTree = {maxIndex_Label: {}}
#     del (label[independ_index])
#
#     selectedCol = [rowVector[independ_index] for rowVector in dataSet]
#     setOfselectedCol = set(selectedCol)
#     for val in setOfselectedCol:
#         copyOflable = label[:]
#         subDataSet = splitDataSet(dataSet, independ_index, val)
#         dcTree[maxIndex_Label][val] = prunningDecisionTree_byChaiSquare(subDataSet, copyOflable, t)
#     return dcTree

def prunningDecisionTree_byChaiSquare(dataSet, labelCopy, chaiTest_list, t):
    Y = [rowVector[-1] for rowVector in dataSet]
    label = labelCopy[:]
    copy_chaiTestList = chaiTest_list[:]
    # print(copy_chaiTestList)
    # return the leave node
    # if Y can only be 1 or 0, return the value, no need to split data


    if Y.count(Y[0]) == len(Y):
        return Y[0]

    if len(dataSet[0]) == 1:
        countYEqualTo1 = sum(Y)
        ans = 1 if countYEqualTo1 >= (len(Y) / 2) else 0
        return ans

    if max(copy_chaiTestList) < t:
        # print('terminate at ' + label[0])
        countYEqualTo1 = sum(Y)
        ans = 1 if countYEqualTo1 >= (len(Y) / 2) else 0
        return ans


    independ_index = copy_chaiTestList.index(max(copy_chaiTestList))
    maxIndex_Label = label[independ_index]
    dcTree = {maxIndex_Label: {}}
    del (label[independ_index])
    del (copy_chaiTestList[independ_index])

    selectedCol = [rowVector[independ_index] for rowVector in dataSet]
    setOfselectedCol = set(selectedCol)
    for val in setOfselectedCol:
        copyOflable = label[:]
        subDataSet = splitDataSet(dataSet, independ_index, val)
        dcTree[maxIndex_Label][val] = prunningDecisionTree_byChaiSquare(subDataSet, copyOflable, copy_chaiTestList, t)
    return dcTree

def chaiSquare_list(dataSet):
    ans = [cal_chaiSquare(dataSet, i) for i in range(len(dataSet[0]) - 1)]
    return ans

# def prunningDecisionTree_byDepth(dataSet, labelCopy):
#     Y = [rowVector[-1] for rowVector in dataSet]
#     label = labelCopy[:]
#     # return the leave node
#     # if Y can only be 1 or 0, return the value, no need to split data
#     if Y.count(Y[0]) == len(Y):
#         return Y[0], 0
#
#     if len(dataSet[0]) == 1:
#         countYEqualTo1 = sum(Y)
#         ans = 1 if countYEqualTo1 >= (len(Y) / 2) else 0
#         return ans, 0
#
#     maxIndexOfInfoGain = maximumInformationGainIndex(dataSet)
#     maxIndex_Label = label[maxIndexOfInfoGain]
#     dcTree = {maxIndex_Label: {}}
#     del (label[maxIndexOfInfoGain])
#
#     selectedCol = [rowVector[maxIndexOfInfoGain] for rowVector in dataSet]
#     setOfselectedCol = set(selectedCol)
#     max_depth = -1
#     for val in setOfselectedCol:
#         copyOflable = label[:]
#         subDataSet = splitDataSet(dataSet, maxIndexOfInfoGain, val)
#         dcTree[maxIndex_Label][val], temp_depth = prunningDecisionTree_byDepth(subDataSet, copyOflable)
#         max_depth = max(temp_depth, max_depth)
#     return dcTree, max_depth + 1

def calculateError(tree, mrows):
    tError = 0
    newDataSet, labels = generateDataSet(mrows)
    for rowVector in newDataSet:
        tError += isError(tree, rowVector)
    return float(tError) / mrows

def isError(tree, rowVector):
    k = len(rowVector) - 1
    lables = ['X' + str(i) for i in range(k)]

    while tree != 1 and tree != 0:
        label = list(tree.keys())
        index = lables.index(label[0])
        value = rowVector[index]
        if value in tree[label[0]]:
            tree = tree[label[0]][value]
        else: return 1
    return 0 if tree == rowVector[-1] else 1

def cal_typicalError(m, sampleSize):
    ave_error = 0
    for i in range(sampleSize):
        dataSet, labels = generateDataSet(m)
        tree = buildDecisionTree(dataSet, labels)
        ave_error += cal_typicalError(tree, 100)
    return float(ave_error) / sampleSize

def calculateTrainError(tree, dataSet):
    trainError = 0
    for rowVector in dataSet:
        trainError += isError(tree, rowVector)
    return float(trainError) / len(dataSet)

def calculateTestError(tree):
    # create 2000 test points
    return calculateError(tree, 2000)

def isContainIrrelevantVariable(tree, lables, sum):
    if tree == 1 or tree == 0:
        return sum
    lable = list(tree.keys())[0]
    index = lables.index(lable)
    if index > 14 and index < 21 and (lable not in sum):
        sum.append(lable)

    # left = isContainIrrelevantVariable(tree[lable][0], lables, sum)
    # right = isContainIrrelevantVariable(tree[lable][1], lables, left)
    for val in list(tree[lable].keys()):
        sum = isContainIrrelevantVariable(tree[lable][val], lables, sum)
    return sum

def averageNumberOfInrrelevantVar(testNumber, m):
    sum = 0
    for i in range(testNumber):
        dataSet, lables = generateDataSet(m)
        tree = buildDecisionTree(dataSet, lables)
        sum += len(isContainIrrelevantVariable(tree, lables, []))
    return float(sum) / testNumber

def avgInrrVar_byDepth(testNumber, m):
    sum = 0
    selectDepth = 10
    for i in range(testNumber):
        dataSet, lables = generateDataSet(m)
        tree = prunningDecisionTree_byDepth(dataSet, lables, 0, selectDepth)
        sum += len(isContainIrrelevantVariable(tree, lables, []))
    return float(sum) / testNumber

def avgInrrVar_bySize(testNumber, m):
    sum = 0
    selectSize = 12
    for i in range(testNumber):
        dataSet, lables = generateDataSet(m)
        tree = prunningDecisionTree_bySize(dataSet, lables, selectSize)
        sum += len(isContainIrrelevantVariable(tree, lables, []))
    return float(sum) / testNumber

def avgInrrVar_byChaiSquare(testNumber, m):
    sum = 0
    alpha = 3.841
    for i in range(testNumber):
        dataSet, lables = generateDataSet(m)
        tlist = chaiSquare_list(dataSet)
        tree = prunningDecisionTree_byChaiSquare(dataSet, lables, tlist, alpha)
        sum += len(isContainIrrelevantVariable(tree, lables, []))
    return float(sum) / testNumber

def cal_chaiSquare(dataSet, index):
    Y = [row[-1] for row in dataSet]
    X = [row[index] for row in dataSet]
    subDataSet = [[x, y] for x, y in zip(X, Y)]
    N = len(dataSet)
    # O = [O_00, O_01, O_10, O_11]
    O = [0] * 4
    # E = [E_00, E_01, E_10, E_11]
    E = [0] * 4
    for row in subDataSet:
        # O_00
        if row[0] == 0 and row[1] == 0:
            O[0] += 1
        elif row[0] == 0 and row[1] == 1:
            O[1] += 1
        elif row[0] == 1 and row[1] == 0:
            O[2] += 1
        else:
            O[3] += 1

    E[0] = (float(O[0] + O[1]) / N) * (float(O[0] + O[2]) / N) * N
    E[1] = (float(O[0] + O[1]) / N) * (float(O[1] + O[3]) / N) * N
    E[2] = (float(O[2] + O[3]) / N) * (float(O[0] + O[2]) / N) * N
    E[3] = (float(O[2] + O[3]) / N) * (float(O[1] + O[3]) / N) * N

    # cal T
    T = 0
    for a, b in zip(O, E):
        if b == 0:
            continue
        T += (a - b) ** 2 / b
    return T


if __name__=='__main__':
    # ----------- Question 1 --------------
    dataSet, labels = generateDataSet(10000)
    tree = buildDecisionTree(dataSet, labels)
    print('dataset looks like this:')
    print(dataSet)
    print(labels)
    print('The decision tree by ID3 looks like this:')
    print(tree)

    print('given decision tree t, calculate error(f):')
    print(calculateError(tree, 10000))

    print('plot the typical error on m data points as m function:')
    m_list = [i for i in range(100, 2050, 50)]
    typical_error = [cal_typicalError(m) for m in m_list]
    plt.plot(m_list, typical_error)
    plt.title('typical error in terms of m')
    plt.xlabel('m-value')
    plt.ylabel('typical error')
    plt.show()

    # ---------- Question 2 ---------------
    # m_list = [1000, 5000, 10000, 50000, 100000]
    # y = [averageNumberOfInrrelevantVar(20, m) for m in m_list]
    # plt.plot(m_list, y)
    # plt.title('Numbers of irrelevant variable')
    # plt.xlabel('m-value')
    # plt.ylabel('Average numbers of irrelevant variabls')
    # plt.show()

    # --------- Question 3a ---------------
    # dataSet, lables = generateDataSet(8000)
    # maxDepth = [i for i in range(50)]
    # train_error = []
    # test_error = []
    # for depth in maxDepth:
    #     tree = prunningDecisionTree_byDepth(dataSet, lables, 0, depth)
    #     train_error.append(calculateTrainError(tree, dataSet))
    #     test_error.append(calculateTestError(tree))
    # xminorLocator = MultipleLocator(5)
    # plt.plot(maxDepth, train_error, color='black', label='$train error$', linewidth=0.8)
    # plt.plot(maxDepth, test_error, color='red', label='$test error$', linewidth=0.8)
    # plt.xlabel('The max depth of the decision tree')
    # plt.ylabel('Error')
    # plt.legend(loc='upper right', frameon=False)
    # plt.show()

    # --------- Question 3b ----------------
    # dataSet, lables = generateDataSet(8000)
    # maxSize = [i for i in range(22)]
    # train_error = []
    # test_error = []
    #
    # for depth in maxSize:
    #     tree = prunningDecisionTree_bySize(dataSet, lables, depth)
    #     train_error.append(calculateTrainError(tree, dataSet))
    #     test_error.append(calculateTestError(tree))
    # plt.plot(maxSize, train_error, color='black', label='$train error$', linewidth=0.8)
    # plt.plot(maxSize, test_error, color='red', label='$test error$', linewidth=0.8)
    # plt.xlabel('The max remaining size of the decision tree')
    # plt.ylabel('Error')
    # plt.legend(loc='upper right', frameon=False)
    # plt.show()

    # -------- Question 3c ---------------
    # dataSet, lables = generateDataSet(8000)
    # tlist = chaiSquare_list(dataSet)
    # alpha = [1.642, 2.706, 3.170, 3.841, 5.024, 6.635, 7.879, 10.828, 12.116, 20, 40]
    #
    # train_error = []
    # test_error = []
    #
    # for t in alpha:
    #     tree = prunningDecisionTree_byChaiSquare(dataSet, lables, tlist, t)
    #     train_error.append(calculateTrainError(tree, dataSet))
    #     test_error.append(calculateTestError(tree))
    # plt.plot(alpha, train_error, color='black', label='$train error$', linewidth=0.8)
    # plt.plot(alpha, test_error, color='red', label='$test error$', linewidth=0.8)
    # plt.xlabel('The significance level alpha')
    # plt.ylabel('Error')
    # plt.legend(loc='upper right', frameon=False)
    # plt.show()

    # -------- Question 5 ----------------
    # m_list = [1000, 5000, 10000, 50000, 100000]
    # y_prunning = [avgInrrVar_byDepth(10, m) for m in m_list]
    # y_origin = [averageNumberOfInrrelevantVar(10, m) for m in m_list]
    # plt.plot(m_list, y_prunning, color='red', label='$Prunning_ByMaxDepth$', linewidth=0.8)
    # plt.plot(m_list, y_origin, color='black', label='$NotPrunning$', linewidth=0.8)
    # plt.xlabel('m-value')
    # plt.ylabel('Average Numbers of Irrelevant Variable')
    # plt.legend(loc='upper right', frameon=False)
    # plt.show()

    # -------- Question 6 ----------------
    # m_list = [1000, 5000, 10000, 50000, 100000]
    # y_prunning = [avgInrrVar_bySize(10, m) for m in m_list]
    # y_origin = [averageNumberOfInrrelevantVar(10, m) for m in m_list]
    # plt.plot(m_list, y_prunning, color='red', label='$PrunningBySize$', linewidth=0.8)
    # plt.plot(m_list, y_origin, color='black', label='$NotPrunning$', linewidth=0.8)
    # plt.xlabel('m-value')
    # plt.ylabel('Average Numbers of Irrelevant Variable')
    # plt.legend(loc='upper right', frameon=False)
    # plt.show()

    #-------- Question 7 -----------------
    # m_list = [1000, 5000, 10000, 50000, 100000]
    # y_prunning = [avgInrrVar_byChaiSquare(10, m) for m in m_list]
    # y_origin = [averageNumberOfInrrelevantVar(10, m) for m in m_list]
    # plt.plot(m_list, y_prunning, color='red', label='$PrunningByChaiSquare$', linewidth=0.8)
    # plt.plot(m_list, y_origin, color='black', label='$NotPrunning$', linewidth=0.8)
    # plt.xlabel('m-value')
    # plt.ylabel('Average Numbers of Irrelevant Variable')
    # plt.legend(loc='upper right', frameon=False)
    # plt.show()











