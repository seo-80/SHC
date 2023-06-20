import numpy as np
import matplotlib.pyplot as plt
import random
import time
from scipy.integrate import odeint
randam = np.random.default_rng()

class shc():
    def __init__(self,kappa,rambda,var,rho=False,x0=[0,0,0]) -> None:
        self.kappa=kappa
        self.rambda=rambda
        self.var=var
        if rho:
            self.rho=rho
        else:
            self.rho=rho(range(len(x0)))
        self.x=np.array(x0)

    def differentiate(self,v,x="empty"):
        if x=="empty":
            return self.kappa*(-self.rambda*self.x-self.rho(v)@S(self.x))
        else:  
            return self.kappa*(-self.rambda*self.x-self.rho(v)@S(x))
    def update(self,v):
        self.x=self.x+self.differentiate(v)+randam.normal(scale=self.var[0])
    def get_x(self):
        return self.x
    def get_v(self):
        return S(self.x)+randam.normal(scale=self.var[1])

class rho:
    def __init__(self,sequenses) -> None:
        self.R=np.array([[[5.0 for i in sequenses[0]] for j in sequenses[0]] for k in sequenses])
        for k in range(len(sequenses)):
            for i in range(len(sequenses[0])):
                for j in range(len(sequenses[0])):
                    if i==j:
                        self.R[(k,i,j)]=1
                    elif sequenses[k].index(i)-1==sequenses[k].index(j):
                        self.R[(k,i,j)]=0.5
                    elif i==sequenses[k][0] and j==sequenses[k][-1]:
                        self.R[(k,i,j)]=0.5
    def __call__(self, vvec=[50]):
        self.ret=np.zeros(self.R[0].shape[0])
        for i,v in enumerate(vvec):
            self.ret=self.ret+self.R[i]*v
        return self.ret

def S(x,G0=50,beta=0.5):
    return G0/(1+np.exp(-1*beta*x))
def S_diff(x):
    return (50-S(x))*S(x)/100


class speaker():
    def __init__(self,x0=False,var=[[0.1,0.1],[0.1,0.1]]) -> None:

        self.var=var
        if  not x0:
            x0=[np.array([[random.randint(-30,0)],[random.randint(-30,0)],[random.randint(-30,0)]]),np.array([[random.randint(-30,0)],[random.randint(-30,0)],[random.randint(-30,0)],[random.randint(-30,0)]])]
        self.x2=(np.array(x0[0]))
        self.x1=(np.array(x0[1])) 
        self.v2=S(self.x2)
        self.v1=S(self.x1)
        self.rho2=rho([[0,1,2],[2,1,0],[1,2,0]])
        self.rho1=rho([[0,1,2,3],[3,2,1,0],[1,2,3,0]])
        self.shc2=shc(1/32,0.3,var[0],self.rho2,self.x2)
        self.shc1=shc(1/8,0.3,var[1],self.rho1,self.x1)
    def update(self):
        self.shc2.update([1])
        self.shc1.update(self.shc2.get_v())
    def simulate(self,n):
        for i in range(n-1):
            self.update()
            self.x2=np.append(self.x2,self.shc2.get_x().reshape(3,1),axis=1)
            self.x1=np.append(self.x1,self.shc1.get_x().reshape(4,1),axis=1)
            self.v2=np.append(self.v2,self.shc2.get_x().reshape(3,1),axis=1)
            self.v1=np.append(self.v1,self.shc1.get_x().reshape(4,1),axis=1)

class recognizer():
    def __init__(self, x0=False,var=[[0.1,0.1],[0.1,0.1]],use_scipy=True) -> None:
        self.var=var
        self.x2=np.array([random.randint(-30,0),random.randint(-30,0),random.randint(-30,0)])
        self.x1=np.array([random.randint(-30,0),random.randint(-30,0),random.randint(-30,0),random.randint(-30,0)])
        self.v2=S(self.x2)
        self.v1=S(self.x1)
        self.x2_record=self.x2.reshape(3,1)
        self.x1_record=self.x1.reshape(4,1)
        self.v2_record=self.v2.reshape(3,1)
        self.v1_record=self.v1.reshape(4,1)
        self.rho2=rho([[0,1,2],[2,1,0],[1,2,0]])
        self.rho1=rho([[0,1,2,3],[3,2,1,0],[1,2,3,0]])
        self.shc2=shc(1/32,0.3,[0,0],self.rho2,self.x2)
        self.shc1=shc(1/8,0.3,[0,0],self.rho1,self.x1)
        self.var_scaler=1#2にする。計算速度見たくてオーバフローしちゃう時に0とかにする
        self.x2_old=self.x2
        self.v2_old=self.v2
        self.x1_old=self.x1
        self.use_scipy=use_scipy


    def update(self,v1,h):
        self.shc2.x=self.x2
        self.x2=self.x2+(self.shc2.differentiate([1])+(self.v2-S(self.x2))*S_diff(self.x2)/(self.var[0][1]**self.var_scaler)-(self.x2-(self.x2_old+self.shc2.differentiate([1],self.x2_old)))/(self.var[0][0]**self.var_scaler))*h

        self.shc1.x=self.x1
        f1_diff=False
        f1_diff=np.array([self.rho1([1,0,0])@S(self.x1)/8])
        f1_diff=np.append(f1_diff,[self.rho1([0,1,0])@S(self.x1)/8],axis=0)
        f1_diff=np.append(f1_diff,[self.rho1([0,0,1])@S(self.x1)/8],axis=0)
        # f1_diff=np.array([self.rho1([1,0,0])@S(self.x1)/8,self.rho1([0,1,0])@S(self.x1)/8,self.rho1([0,0,1])@S(self.x1)/8])
        
        self.v2=self.v2+(S(self.x2-self.x2_old)    +f1_diff@(self.x1-self.x1_old-self.shc1.differentiate(self.v2))/(self.var[1][0]**self.var_scaler)-(self.v2-S(self.x2))/(var[0][1]**self.var_scaler))*h

        self.x1=self.x1+(self.shc1.differentiate(self.v2)+(self.v1-S(self.x1))*S_diff(self.x1)/(self.var[1][1]**self.var_scaler)-(self.x1-(self.x1_old+self.shc1.differentiate(self.v2)))/(self.var[1][0]**self.var_scaler))*h
        self.v1=v1

        self.x2_old=self.x2
        self.v2_old=self.v2
        self.x1_old=self.x1
    def update_by_scipy(self,v1,Division_number=1):

        self.v1=v1
        f1_diff=np.array([self.rho1([1,0,0])@S(self.x1)/8])
        f1_diff=np.append(f1_diff,[self.rho1([0,1,0])@S(self.x1)/8],axis=0)
        f1_diff=np.append(f1_diff,[self.rho1([0,0,1])@S(self.x1)/8],axis=0)
        t=np.linspace(0, 1, Division_number)
        self.x1=odeint(lambda x,t:(self.shc1.differentiate(self.v2)+(self.v1-S(x))*S_diff(x)/(self.var[1][1]**self.var_scaler)-(x-(self.x1_old+self.shc1.differentiate(self.v2)))/(self.var[1][0]**self.var_scaler)),self.x1,t)[-1]
        self.x2=odeint(lambda x,t:(S(x-self.x2_old)    +f1_diff@(self.x1-self.x1_old-self.shc1.differentiate(self.v2))/(self.var[1][0]**self.var_scaler)-(self.v2-S(x))/(var[0][1]**self.var_scaler)),self.x2   ,t)[-1]
        self.v2=odeint(lambda x,t:(S(self.x2-self.x2_old)    +f1_diff@(self.x1-self.x1_old-self.shc1.differentiate(x))/(self.var[1][0]**self.var_scaler)-(x-S(self.x2))/(var[0][1]**self.var_scaler)),self.v2,t)[-1]
        self.x2_old=self.x2
        self.v2_old=self.v2
        self.x1_old=self.x1
    def simulate(self,v1,n,Division_number=1):
        for i in range(n-1):
            if i%10==0:
                print(i)
                print(self.x1)
            if self.use_scipy:
                self.update_by_scipy(v1[:,i],Division_number=Division_number)
            else:
                for j in range(Division_number):
                    self.update(v1[:,i],1/Division_number)
            self.x2_record=np.append(self.x2_record,self.x2.reshape(3,1),axis=1)
            self.x1_record=np.append(self.x1_record,self.x1.reshape(4,1),axis=1)
            self.v2_record=np.append(self.v2_record,self.v2.reshape(3,1),axis=1)
            self.v1_record=np.append(self.v1_record,self.v1.reshape(4,1),axis=1)


n=1000
var=[[0.1,0.1],[0.1,0.1]]
speaker=speaker(var=var)
recognizer=recognizer(var=var)
speaker.simulate(n)
start=time.time()
recognizer.simulate(speaker.v1,n,Division_number=100000)#分割数は10000は必要
print(time.time()-start)




figure_num=6
plt.subplot(figure_num*100+11)
for i in range(3):
    plt.plot(range(n),speaker.x2[i])
plt.subplot(figure_num*100+12)
for i in range(4):
    plt.plot(range(n),speaker.x1[i])
plt.subplot(figure_num*100+13)
for i in range(4):
    plt.plot(range(n),recognizer.x1_record[i])
plt.subplot(figure_num*100+14)
for i in range(3):
    plt.plot(range(n),recognizer.v2_record[i])
plt.subplot(figure_num*100+15)
for i in range(3):
    plt.plot(range(n),recognizer.x2_record[i])
plt.subplot(figure_num*100+16)
for i in range(3):
    plt.plot(range(n),recognizer.x2_record[i]-speaker.x2[i])

plt.show()
