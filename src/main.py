import numpy as np
import matplotlib.pyplot as plt

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




rho2=rho([[0,1,2],[2,1,0],[1,2,0]])
rho1=rho([[0,1,2,3],[3,2,1,0],[1,2,3,0]])



n=5000

x2=S(np.array([[-25],[-20],[-10]]))
x1=S(np.array([[-25],[-20],[-10],[-20]]))
shc2=shc(1/8,0.3,rho2,x2)
shc1=shc(1/2,0.3,rho1,x1)
for i in range(n-1):
    shc2.update([1])
    shc1.update(shc2.get_v())
    x2=np.append(x2,shc2.get_v().reshape(3,1),axis=1)
    x1=np.append(x1,shc1.get_v().reshape(4,1),axis=1)
plt.subplot(211)
for i in range(3):
    plt.plot(range(n),x2[i])
plt.subplot(212)
for i in range(4):
    plt.plot(range(n),x1[i])

plt.show()