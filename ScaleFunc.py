from qutip import *
import numpy as np
from pylab import *

N_a = 2 

def ScaleUpOp(x, N_a):
    # create an empty subspace of the operator   
    BaseArray = Qobj(np.zeros(shape(x)),dims(x))
    # tensor that up to the level of N_a atoms
    for i in range(1, N_a):
        BaseArray = tensor(BaseArray, qeye(N))
#    print(BaseArray)
 # now take the operator and tensor it up for each atom
    for i in range(N_a):
        if i == 0:
            SubspaceObj = x
            for j in range(1, N_a):
                    SubspaceObj = tensor(SubspaceObj,qeye(N))
#            print(SubspaceObj)
            BaseArray = Qobj(BaseArray.data + SubspaceObj.data)
#            print(BaseArray)
        else:
            SubspaceObj = qeye(N)
            for j in range(1, N_a):
                if j == i:
                    SubspaceObj = tensor(SubspaceObj, x)
                else:
                    SubspaceObj = tensor(SubspaceObj, qeye(N))
#            print(SubspaceObj)
            BaseArray = Qobj(BaseArray.data + SubspaceObj.data)
#            print(BaseArray)
    return BaseArray