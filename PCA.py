from numpy import *
import matplotlib.pyplot as plt

def loadDataSet(fileName,delim='\t'):
    fr=open(fileName)
    stringArr=[line.strip().split(delim) for line in fr.readlines()]
    datArr=[list(map(float,line)) for line in stringArr]
    return mat(datArr)

def pca(dataMat,topNfeat=9999999):
    meanVals=mean(dataMat,axis=0)
    meanRemoved=dataMat-meanVals
    covMat=cov(meanRemoved,rowvar=0)
    eigVals,eigVects=linalg.eig(mat(covMat))
    eigValInd=argsort(eigVals)
    eigValInd=eigValInd[:-(topNfeat+1):-1]
    redEigVects=eigVects[:,eigValInd]
    lowDDataMat=meanRemoved*redEigVects
    reconMat=(lowDDataMat*redEigVects.T)+meanVals
    return lowDDataMat,reconMat


if __name__ == "__main__":
    fr=open("pca.txt")
    stringArr = [line.strip().split("\t") for line in fr.readlines()]
    print(stringArr)
    datArr = mat([list(map(float, line)) for line in stringArr])
    print(datArr)
    meanVals=mean(datArr,axis=0)
    meanRemoved=datArr-meanVals
    print(meanRemoved)
    covMat=cov(meanRemoved,rowvar=0)
    print(covMat)
    eigVals,eigVects=linalg.eig(mat(covMat))
    print(eigVals)
    print(eigVects)
    eigValInd=argsort(eigVals)
    print(eigValInd)

    eigValInd=eigValInd[:-2:-1]
    print(eigValInd)
    redEigVects=eigVects[:,eigValInd]
    print(redEigVects)
    lowDDataMat=meanRemoved*redEigVects
    print(lowDDataMat)
    print(lowDDataMat*redEigVects.T+meanVals)
    print(redEigVects.T)
    print(meanVals)

    # dataMat=loadDataSet("testSetPCA.txt")
    # lowDMat,reconMat=pca(dataMat,2)
    # fig=plt.figure()
    # ax=fig.add_subplot(111)
    # ax.scatter(dataMat[:,0].flatten().A[0],dataMat[:,1].flatten().A[0],marker="^",s=90)
    # ax.scatter(reconMat[:, 0].flatten().A[0], reconMat[:, 1].flatten().A[0], marker="o", s=50,c="red")
    # plt.show()


    # dataMat = loadDataSet("testSetPCA.txt")
    # print(dataMat[:,0].flatten().A[0])
