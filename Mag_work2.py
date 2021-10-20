#importLib
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import csv


def extract_cluster(densSortArr, closestNodeIdArr, classNum, gamma):
    # importAllData
    n = densSortArr.shape[0]

    # initializeCategoryOfEachPoints
    lab = np.full((n,), -1)
    corePoints = np.argsort(-gamma)[: classNum]

    # createCenterOfClusterCattegory
    lab[corePoints] = range(len(corePoints))

    # ConvertMassiveToList
    densSortList = densSortArr.tolist()

    # reverseSolrtList
    densSortList.reverse()

    # assignLabelForEachElements
    for nodeId in densSortList:
# findClosestsPoints


if (lab[nodeId] == -1):
    lab[nodeId] = lab[closestNodeIdArr[nodeId]]
return corePoints, lab
if __name__ == '__main__':
    data = np.loadtxt("/content/VT-11_new.csv", delimiter=",")

    # initClusterAlgorithm
    densityArr, densitySortArr, closestDisOverSelfDensity, closestNodeIdArr, gamma = CFSFDS(data, 2)

    # determineClusterCenter
    G = []
    G = numpy.matrix.tolist(gamma)
    G.sort(reverse=True)
    K = []
    for i in range(len(G) - 1):
        k = G[i] - G[i + 1]
        K.append(k)
    ksum = 0

    for i in range(len(K)):
        ksum = ksum + K[i]
    R = ksum / len(K)
    Result = 1

    for i in range(len(K)):
        if K[i] > R:
            Result = Result + 1

    # calculateClusterCenter
    classNum = Result
    corePoints, labels = extract_cluster(densitySortArr,
                                         closestNodeIdArr,
                                         classNum, gamma)
    # exportClusterCenterCoordinate
    X = data[corePoints, 0]
    Y = data[corePoints, 1]
    CC = []

    for i in range(len(X)):
        CC.append([X[i], Y[i]])

    name1 = ['lat', 'lon']
    S = pd.DataFrame(columns=name1, data=CC)
    # ClusterCenter
    S.to_csv("/content/CC_new.csv", encoding='utf-8')

    # exportClasssificationResults
    M = []
    R = []
    L = []

    for n in corePoints:
        M.append(n)
    with open("/content/VT-12_new.csv", 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
    for i in labels:
        L.append((i))
    for x in range(0, len(rows)):
        R.append([rows[x][0], rows[x][1], L[x]])

    name = ['lat', 'lon', 'code']
    SortR = pd.DataFrame(columns=name, data=R)
    # ClusteringResult
    SortR.to_csv("/content/ResultCenter_new.csv", encoding='utf-8')

#createGraph
import seaborn as sns
import pandas as pd
import matplotlib as plt

datait = pd.read_csv('/content/ResultCenter_new.csv')
df = pd.DataFrame(datait, columns=['lat', 'lon', 'code'])

datait[['lat','lon']].plot(
    kind='scatter',
    x='lat',
    y='lon',
    figsize=(12,8)
)



