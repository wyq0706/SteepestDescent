import numpy as np
import matplotlib.pyplot as plt

INFINITE=2147483647
EPS=1e-10

Fletcher_Reeves=1
Polak_Ribiere=2

filePrefix="D://2017011991//source//optimizer//共轭梯度"


'''
以下函数和求导运算式根据具体问题修改
'''
def func(X):
    x1=X[0]
    x2=X[1]
    return (1-x1)*(1-x1)+2*(x2-x1*x1)*(x2-x1*x1)

def GradientValueForFunc(X):
    x1=X[0]
    x2=X[1]
    return (-2)*(1-x1)+4*(x2-x1*x1)*(-2)*x1,4*(x2-x1*x1)

def OnceGradientForT(X,D,t):
    x1=X[0]
    x2=X[1]
    d1=D[0]
    d2=D[1]
    return (-2)*(1-x1-t*d1)*d1+4*(x2+t*d2-np.power(x1+t*d1,2))*(d2-2*d1*(x1+t*d1))

def twiceGradientForT(X,D,t):
    x1=X[0]
    x2=X[1]
    d1=D[0]
    d2=D[1]
    return 24*np.power(d1,2)*np.power(x1+t*d1,2)-16*d1*d2*(x1+t*d1)-8*d1*d2*(x2+t*d2)+2*np.power(d1,2)+4*np.power(d2,2)


'''
牛顿法精确搜索
'''
def NewtonSearch(X,D):
    t=float(0.1)
    # print("func",OnceGradientForT(X,D,1),t)
    while np.abs(OnceGradientForT(X,D,t))>EPS:
        t=t-OnceGradientForT(X,D,t)/twiceGradientForT(X,D,t)
        # print("func",OnceGradientForT(X,D,t),t)
    return t


'''
梯度下降精确搜索
'''
def GradientSearch(X,D):
    t=float(0.5)
    # print("func",OnceGradientForT(X,D,1),t)
    while np.abs(OnceGradientForT(X,D,t))>EPS*0.01:
        t=t-OnceGradientForT(X,D,t)
        # print("func",OnceGradientForT(X,D,t),t)
    return t


''' 
最速下降法的梯度方向获取
'''
def GradientForSteepestDescent(X,p):
    gX1,gX2=GradientValueForFunc(X)
    if p==1:
        rtn=np.array([0,0])
        if np.abs(gX1)>np.abs(gX2):
            rtn[0]=np.sign(-1*gX1)
        else:
            rtn[1]=np.sign(-1*gX2)
        return rtn
    elif p==INFINITE:
        return np.array([np.sign(-1*gX1),np.sign(-1*gX2)])
    else:
        q=1/(1-1/p)
        pValue=np.linalg.norm([gX1,gX2],q)
        d1=np.sign(-gX1)*np.power(gX1,q)*np.power(pValue,-q/p)
        d2=np.sign(-gX2)*np.power(gX2,q)*np.power(pValue,-q/p)
        return np.array([d1,d2])



'''
function: 最速下降法
params: p为范数取值
return：函数最优解，最优值，迭代次数
'''
def SteepestDescent(p):
    X=np.array([0,0],dtype=float)
    count=0
    # 画图
    plt1=plt.subplot(121)
    plt2=plt.subplot(122)
    listX=[X[0]]
    listY=[X[1]]
    listVal=[func(X)]
    listC=[0]

    while np.linalg.norm([GradientValueForFunc(X)],2)>EPS:
        D=GradientForSteepestDescent(X,p)
        t=NewtonSearch(X,D)
        X+=D*t
        count+=1

        listX.append(X[0])
        listY.append(X[1])
        listC.append(count)
        listVal.append(func(X))
    plt1.plot(listX,listY)
    plt1.set_xlabel("x1")
    plt1.set_ylabel("x2")
    plt2.plot(listC,listVal)
    plt2.set_ylabel("f(X)")
    plt2.set_xlabel("count")
    plt.savefig(filePrefix+str(p)+".jpg")
    plt.show()
    plt.close()
    return X,func(X),count


'''
function: 共轭梯度法
params: Fletcher_Reeves/Polak_Ribiere
return：函数最优解，最优值，迭代次数
'''
def ConjugateGradient(p):
    X=np.array([0,0],dtype=float)
    count=0
    lastD=np.array([0,0])
    # 画图
    plt1=plt.subplot(121)
    plt2=plt.subplot(122)
    listX=[X[0]]
    listY=[X[1]]
    listVal=[func(X)]
    listC=[0]


    while np.linalg.norm([GradientValueForFunc(X)],2)>EPS:
        if count==0:
            D=(-1)*np.array(GradientValueForFunc(X))
            t=NewtonSearch(X,D)
            X+=D*t
        else:
            D=(-1)*np.array(GradientValueForFunc(X))
            if p==Fletcher_Reeves:
                a=np.power(np.linalg.norm(D,2)/np.linalg.norm(lastD,2),2)
            else:
                a=np.matmul(D,D-lastD)/np.power(np.linalg.norm(lastD,2),2)
            D+=a*lastD
            # Newton does not converge
            t=GradientSearch(X,D)
            X+=D*t
            # print("X",X)

        lastD=D
        count+=1

        listX.append(X[0])
        listY.append(X[1])
        listC.append(count)
        listVal.append(func(X))
    plt1.plot(listX,listY)
    plt1.set_xlabel("x1")
    plt1.set_ylabel("x2")
    plt2.plot(listC,listVal)
    plt2.set_ylabel("f(X)")
    plt2.set_xlabel("count")
    plt.savefig(filePrefix+str(p)+".jpg")
    plt.show()
    plt.close()
    return X,func(X),count


if __name__ == "__main__":
    # 最速下降法
    # l1范数
    rtn, val, cnt=SteepestDescent(1)
    print(rtn[0],rtn[1],val,cnt)

    # l2范数
    rtn, val, cnt=SteepestDescent(2)
    print(rtn[0],rtn[1],val,cnt)

    # l_无穷大 范数
    rtn, val, cnt=SteepestDescent(INFINITE)
    print(rtn[0],rtn[1],val,cnt)

    # 共轭梯度法
    # Fletcher_Reeves
    rtn, val, cnt=ConjugateGradient(Fletcher_Reeves)
    print(rtn[0],rtn[1],val,cnt)

    # Polak_Ribiere
    rtn, val, cnt=ConjugateGradient(Polak_Ribiere)
    print(rtn[0],rtn[1],val,cnt)