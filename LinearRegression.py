from numpy import *
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    numFeat=len(open(fileName).readline().split("\t"))-1
    dataMat=[]
    labelMat=[]
    fr=open(fileName)
    for line in fr.readlines():
        lineArr=[]
        curLine=line.strip().split("\t")
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat


def standRegres(xArr,yArr):
    xMat=mat(xArr)
    yMat=mat(yArr).T
    xTx=xMat.T*xMat
    if linalg.det(xTx)==0:
        print("This matrix is singular,cannot do inverse")
        return
    ws=xTx.I*(xMat.T*yMat)
    return ws



def lwlr(testPoint,xArr,yArr,k=1.0):
    xMat=mat(xArr)
    yMat=mat(yArr).T
    m=shape(xMat)[0]
    weights=mat(eye((m)))
    dM=testPoint-xMat[1,:]
    for j in range(m):
        diffMat=testPoint-xMat[j,:]
        weights[j,j]=exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx=xMat.T*(weights*xMat)
    if linalg.det(xTx)==0.0:
        print("This matrix is singular,cannot do inverse")
        return
    ws=xTx.I*(xMat.T*(weights*yMat))
    # print(ws)
    return testPoint*ws

def lwlrTest(testArr,xArr,yArr,k=1.0):
    m=shape(testArr)[0]
    yHat=zeros(m)
    for i in range(m):
        yHat[i]=lwlr(testArr[i],xArr,yArr,k)
    return yHat



def ridgeRegres(xMat,yMat,lam=0.2):
    xTx=xMat.T*xMat
    denom=xTx+eye(shape(xMat)[1])*lam
    # print(shape(xMat))
    if linalg.det(denom)==0.0:
        print("This matrix is singular,cannot do inverse")
        return
    ws=denom.I*(xMat.T*yMat)
    return ws


def ridgeTest(xArr,yArr):
    xMat=mat(xArr)
    yMat=mat(yArr).T
    # print(xMat)
    yMean=mean(yMat,0)
    # print(yMean)
    yMat=yMat-yMean
    xMeans=mean(xMat,0)
    xVar=var(xMat,0)
    # print(xMeans)
    # print(xVar)
    xMat=(xMat-xMeans)/xVar
    numTestPts=30
    wMat=zeros((numTestPts,shape(xMat)[1]))
    # print(wMat.shape)
    for i in range(numTestPts):
        ws=ridgeRegres(xMat,yMat,exp(i-10))
        wMat[i,:]=ws.T
    return wMat


if __name__=='__main__':
    # xArr,yArr=loadDataSet("ex0.txt")
    # ws=standRegres(xArr,yArr)
    # xMat=mat(xArr)
    # yMat=mat(yArr)
    # fig=plt.figure()
    # ax=fig.add_subplot(111)
    # ax.scatter(xMat[:,1].flatten().A[0],yMat.T[:,0].flatten().A[0])
    # xCopy=xMat.copy()
    # xCopy.sort(axis=0)
    # print(xCopy)
    # yHat=xCopy*ws
    # ax.plot(xCopy[:,1],yHat)
    #plt.show()
    # yHat=xMat*ws
    # ax.scatter(xMat[:,1].flatten().A[0],yMat.T[:,0].flatten().A[0])
    # ax.scatter(yHat,yMat)
    # plt.show()
    # print(corrcoef(yHat.T,yMat))




    # xArr,yArr=loadDataSet("ex0.txt")
    # yHat=lwlrTest(xArr,xArr,yArr,0.003)
    # xMat=mat(xArr)

    # print(xMat[:,1].shape)
    # srtInd=xMat[:,1].argsort(0)
    # print(srtInd)
    # print(xMat)
    # print(srtInd)
    # xSort=xMat[srtInd]
    # print(xSort)


    # fig=plt.figure()
    # ax=fig.add_subplot(111)
    # ax.plot(xSort[:,1],yHat[srtInd])
    # ax.scatter(xMat[:,1].flatten().A[0],mat(yArr).T.flatten().A[0],s=2,c='red')
    # plt.show()

    abX,abY=loadDataSet('abalone.txt')
    ridgeWeights=ridgeTest(abX,abY)


    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.plot(ridgeWeights)
    plt.show()