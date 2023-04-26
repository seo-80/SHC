import numpy as np
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
        print(self.R[0].shape)
        self.ret=np.zeros(self.R[0].shape[0])
        for i,v in enumerate(vvec):
            self.ret=self.ret+self.R[i]*v
        return self.ret
print(rho.R)
rho=rho([[0,1,2,3,4],[4,3,2,1,0],[3,4,1,2,0]])
print(rho())