import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from numba import jit
import random

@jit
def make_d(d,A,N):#次数行列を作成
    for i in range(N):
        for j in range(N):
            if A[i,j]==1:
                d[i,i]=d[i,i]+1


@jit
def max_d(d,N):#最大のノード次数を取得
    #max=0
    min=N
    index=0
    for i in range(N):
        #if d[i][i]>max:
            #max=d[i][i]
            #index=i
        if d[i][i]<min:
            min=d[i][i]
            index=i
    print("index",index)
    #print("max",d[index][index])
    print("min",d[index][index])
    return index


@jit
def adj(N,A,u,index):#隣接行列に初期値を与える
    #for i in range(N):
        #if A[index,i]==1 or i==index:
            #u[i]=0.1
    u[index]=0.1

@jit
def laplacian(s,L):#ラプラシアンを求める
    L1=s.shape
    S=int(s.size)
    ts = np.zeros(S)
    for i in range(S):
        for j in range(S):
            #ts[i]+=L[j,i]*s[j]
            ts[i]+=L[i,j]*s[j]
    return ts


@jit
def calc(a, h, a2, h2, La,c,u,v,fa,fh,ga,gh,roa,roh):#状態量を求める
    L = a.size
    (L2,L2)=La.shape
    heikin=0
    dt=0.01
    Dh=0.5
    Da=0.058+c*0.001
    mina=0
    minh=0
    maxa=1
    maxh=1
    sa = ((fa*a)+(fh*h)+roa -Da * laplacian(a,La))*dt ##反応項と拡散項を計算
    sh = ((ga*a)+roh+(gh*h)-Dh * laplacian(h,La))*dt
    for i in range(L):
            a2[i] = a[i]+(sa[i]) #-mua*a[i,j]
            h2[i] = h[i]+(sh[i]) # -muh*h[i,j]
            if a2[i]<mina:
                a2[i]=mina
            if h2[i]<minh:
                h2[i]=minh
            if a2[i]>maxa:
                a2[i]=maxa
            if h2[i]>maxh:
                h2[i]=maxh



def pic(N,a,h,G,indexlist,us,vs,i):#図示する
    u0=np.zeros(N)
    v0=np.zeros(N)
    u=np.zeros(N)
    v=np.zeros(N)
    judge=0
    for j in range(N):
        u[j]=round(a[j],2)
        v[j]=round(h[j],2)
        u0[j]=round(us[j],2)
        v0[j]=round(vs[j],2)
        if u[j]!=u0[j]or v[j]!=v0[j]:
            judge=1
    return judge

def plot(time_kappa,indexlist):
    N = indexlist.size
    fig, ax = plt.subplots()
    ax.set_title('convergence time')
    for i in range(N):
      plt.scatter(indexlist[i], time_kappa[i],c="red",s=10)
      #plt.scatter(indexlist[i], u0[i],c="black",s=10)
    ax.set_xlabel("kappa_a")
    ax.set_ylabel("time")
    plt.show()



def main():
    N = 200# the number of points
    d=np.zeros((N,N))
    A=np.zeros((N, N))
    np.random.seed(seed=0)
    #レギュラーグラフ
    #################################################
    #G = nx.random_regular_graph(4, N, seed=0)#レギュラーグラフ
    #G=nx.erdos_renyi_graph(N, 0.02,seed=33)
    G = nx.random_regular_graph(4, N, seed=0)
    pos = nx.spring_layout(G)
    A = nx.to_numpy_matrix(G)
    ############################################
    make_d(d,A,N)
    L=d-A#ラプラシアン行列
    ############################################
    ca=0.08
    ch=0.11
    da=0.08
    mua=0.03
    muh=0.12
    roa=np.zeros(N)
    roh=np.zeros(N)
    for i  in range(N):
        roa[i]=(mua-ca+da)*(0.01*L[i,i])
        roh[i]=(-ch+muh)*(0.01*L[i,i])
    fa=ca-mua
    fh=-da
    ga=ch
    gh=-muh
    ############################################
    La=np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            La[i][j]=L[i,j]/L[j,j]
    h=N//2
    u0= np.zeros(N)+0.1
    adj(N,A,u0,h)
    u02 =np.zeros(N)
    v0 = np.zeros(N)+0.1
    v02 =np.zeros(N)
    uc0 =np.zeros(N)
    vc0 =np.zeros(N)
    random.seed(0)
    for i in range(N):#初期状態をランダムに設定
            u0[i] = 0.01*(L[i,i])
            v0[i] = 0.01*(L[i,i])
    #print(L)
    plt.subplot()
    plt.figure(figsize=(6,4))
    nx.draw(G, node_size=20)
    nx.draw(G, node_size=20)
    plt.tight_layout()
    plt.show()
    kizami=int((0.1-0.058)/0.001+2)
    time=100000
    time_kappa=np.zeros(kizami)
    indexlist=np.zeros(kizami)
    for k in range(kizami):
        a=np.zeros(N)
        a2=np.zeros(N)
        h=np.zeros(N)
        h2=np.zeros(N)
        indexlist[k]=0.058+(k)*0.001
        for i in range(N):#これができないと配列が初期化できない
                a[i]=u0[i]
                a2[i]=u02[i]
                h[i]=v0[i]
                h2[i]=v02[i]
                max_index=np.argmax(u0)
                a[max_index]=u0[max_index]+0.03
        for i in range(time):
            if i % 2 == 0:
                    calc(a, h, a2, h2, La,k,u0,v0,fa,fh,ga,gh,roa,roh)
            else:
                    calc(a2, h2, a, h, La,k,u0,v0,fa,fh,ga,gh,roa,roh)
                    #現在のステップの状態u2,v2から次のステップの状態u,vを計算する
                    judge=pic(N,a,h,G,indexlist,u0,v0,i)
                    if judge==0:
                        time_kappa[k]=i
                        print(indexlist[k],i)
                        break

    plot(time_kappa,indexlist)

main=main()
