import numpy as np
import matplotlib.pyplot as plt
import random

class shc():
    def __init__(self,kappa,rambda,rho=False,x0=[0,0,0]) -> None:
        self.kappa=kappa
        self.rambda=rambda
        if rho:
            self.rho=rho
        else:
            self.rho=rho(range(len(x0)))
        self.x=np.array(x0)

    def differentiate(self,v):
        return self.kappa*(-self.rambda*self.x-self.rho(v)@S(self.x))
    def update(self,v):
        self.x=self.x+self.differentiate(v)
    def get_x(self):
        return self.x
    def get_v(self):
        return S(self.x)

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

class recognizer(shc):
    def __init__(self, kappa, rambda, rho=False, x0=[0, 0, 0]) -> None:
        super().__init__(kappa, rambda, rho, x0)
        self.sigma=0.0
    def update_sigma(self):
        self.sigma
class speaker():
    def __init__(self,x0=False) -> None:
        if  x0:
            self.x2=(np.array(x0[0]))
            self.x1=(np.array(x0[1]))           
        else:
            self.x2=(np.array([[random.randint(-30,0)],[random.randint(-30,0)],[random.randint(-30,0)]]))
            self.x1=(np.array([[random.randint(-30,0)],[random.randint(-30,0)],[random.randint(-30,0)],[random.randint(-30,0)]]))
        self.v2=S(self.x2)
        self.v1=S(self.x1)
        self.rho2=rho([[0,1,2],[2,1,0],[1,2,0]])
        self.rho1=rho([[0,1,2,3],[3,2,1,0],[1,2,3,0]])
        self.shc2=shc(1/8,0.3,self.rho2,self.x2)
        self.shc1=shc(1/2,0.3,self.rho1,self.x1)
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
        

n=1000
speaker=speaker()
speaker.simulate(n)
plt.subplot(211)
for i in range(3):
    plt.plot(range(n),speaker.v2[i])
plt.subplot(212)
for i in range(4):
    plt.plot(range(n),speaker.v1[i])

plt.show()