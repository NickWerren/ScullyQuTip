from qutip import *
import matplotlib.pyplot as plt
import numpy as np
from pylab import *
from qutip import tensor, basis, destroy, qeye, mesolve
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation

# parameters for total system

# define the atom
N_a = 2 # number of atoms
N_l = 3 # number of levels

# atom 1 states
astate = basis(N_l,0)
bstate = basis(N_l,1)
cstate = basis(N_l,2)

# define the field
N=3 # number of Fock states in Hilbert space of mode 1.

basis_mode_1 = basis(N,0)

Two_Mode = False

if Two_Mode is True:
    M = 3
    basis_mode_2 = basis(M,0)
    # 2 mode field operators
    a_1 = tensor(destroy(N), qeye(M))    
    b_1 = tensor(qeye(N), destroy(M))
    a = tensor(a_1 +b_1, ScaleUpOp(qeye(N_l)))
    I_f = tensor(qeye(N), qeye(M))
    
if Two_Mode is False:
    a = tensor(destroy(N), ScaleUpOp(qeye(N_l)))
    I_f = qeye(N)

# atomic operators
saa = tensor(I_f, astate*astate.dag())
sbb = tensor(I_f, bstate*bstate.dag())
scc = tensor(I_f, cstate*cstate.dag())
sab = tensor(I_f, astate*bstate.dag())
sac = tensor(I_f, astate*cstate.dag())
sbc = tensor(I_f, bstate*cstate.dag())
I_single = tensor(I_f, qeye(3))
sm = tensor(I_f, destroy(3))
sz_ac_ab = ((saa-scc)+(saa-sbb))
sz_ac_ab_d = ((scc-saa)+(sbb-saa))
sz_bc = (sbb-scc)
sz_bc_d = (scc-sbb)

def ScaleUpOp(x):
    BaseArray = Qobj(np.zeros(shape(x)),dims(x))
    # tensor that up to the level of N_a atoms
    for i in range(1, N_a):
        BaseArray = tensor(BaseArray, qeye(N))
#    print(BaseArray)
    # create an empty subspace of the operator    # now take the operator and tensor it up for each atom
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

# Scaled atomic operators
Saa = tensor(I_f, ScaleUpOp(astate*astate.dag()))
Sbb = tensor(I_f, ScaleUpOp(bstate*bstate.dag()))
Scc = tensor(I_f, ScaleUpOp(cstate*cstate.dag()))
Sab = tensor(I_f, ScaleUpOp(astate*bstate.dag()))
Sac = tensor(I_f, ScaleUpOp(astate*cstate.dag()))
Sbc = tensor(I_f, ScaleUpOp(bstate*cstate.dag()))
I = tensor(I_f, qeye(N_l))
Sm = tensor(I_f, destroy(N_l))
Sz_ac_ab = ((Saa-Scc)+(Saa-Sbb))
Sz_ac_ab_d = ((Scc-Saa)+(Sbb-Saa))
Sz_bc = (Sbb-Scc)
Sz_bc_d = (Scc-Sbb)

# transition frequencies
om_a = 0
om_b = 0
om_c = 0

# laser detuning
nu1 = 0
nu2 = nu1

# ab and ac detunings from cavity
om_ac = (om_a - om_c)
om_ab = (om_a - om_b)
Delta_1 = nu1 - om_ac
Delta_2 = nu2 - om_ab

# bc detuning
om_bc = (om_b - om_c)
Delta_3 = nu2 - om_bc
#bc_detun = np.exp(1j*((om_bc)*t))

# cavity field
Om1 = 0.0 #cavity frequency 1
Om2 = 0.0 #cavity frequency 2
Delta_c1 = Om1 - nu1 # photon detuning
Delta_c2 = Om2 - nu2 # photon detuning

# ab and ac cavity couplings
g1 = 1.0
g2 = 1.0

# rotating wave approximation  #WRONG!
RWA = True
if RWA is False:
    RWA_switch=1
if RWA is True:
    RWA_switch=0
    
xc = (a + RWA_switch*a.dag())
nc = a.dag() * a

# atom Hamiltonian
H_a = (om_a*Saa + om_b*Sbb + om_c*Scc)

# field Hamiltonian
H_f = Om1*a.dag()*a

# atom-field interaction hamiltonian
Hc_ac_m = g2*xc.dag()*(Sab)  # b -> a, a-dag & sigma minus, time-dependent term
def Hc_ac_m_coeff(t, args):
    ab_detun = np.exp(1j*((Delta_1)*t))
    return ab_detun

Hc_ab_m = g1*xc.dag()*(Sac)  # c -> a, a-dag & sigma minus, time-dependent term
def Hc_ab_m_coeff(t, args):
    ac_detun = np.exp(1j*((Delta_2)*t))
    return ac_detun

Hc_ac_p = g2*xc*(Sab.dag())  # a -> b, a & sigma plus, time-dependent term
def Hc_ac_p_coeff(t, args):
    ab_detun = np.exp(-1j*((Delta_1)*t))
    return ab_detun

Hc_ab_p = g1*xc*(Sac.dag())  # a -> c, a & sigma plus, time-dependent term
def Hc_ab_p_coeff(t, args):
    ac_detun = np.exp(-1j*((Delta_2)*t))    
    return  ac_detun

# atom-cavity Hamiltonian
phi = 0
field_rot = np.exp(1j*phi)
H_m = (field_rot*Sbc.dag() +field_rot.conjugate()*Sbc)



scully_hamiltonian = [H_a +H_m +H_f, \
                     [Hc_ac_m,Hc_ac_m_coeff], \
                     [Hc_ab_m,Hc_ab_m_coeff], \
                     [Hc_ac_p,Hc_ac_p_coeff], \
                     [Hc_ab_p,Hc_ab_p_coeff]]

print(scully_hamiltonian)

def collapse_ops(i):
    # build collapse operators
    c_ops = []   
    
    # incoherent pumping
    lam = i # incoherent pumping rate to a from b and c
    c_ops.append(np.sqrt(lam) * Sab.dag()) # a->b
    c_ops.append(np.sqrt(lam) * Sac.dag()) # a->c
        
    # cavity relaxation 
    kappa=0.01 # cavity relaxation rate
    n_th= 0.0 # bath temperature
    c_ops.append(np.sqrt(kappa * (1+n_th)) * a)
    c_ops.append(np.sqrt(kappa * n_th) * a.dag())
    
    # atomic relaxation 
    # cavity decay rates
    Gam1=0.01 # b -> a decay rate
    Gam2=0.01 # c -> a decay rate
    c_ops.append(np.sqrt(Gam1) * Sab)
    c_ops.append(np.sqrt(Gam2) * Sac)

    # ab and ac lifetime and dephasing
    T1 = 1.0
    T2 = 1.0
    c_ops.append(0.5*T1*sz_ac_ab)
    c_ops.append(0.5*T2*sz_ac_ab_d)
    
    # bc lifetime and dephasing
    tau1 = 1.0
    tau2 = 1.0
    c_ops.append(0.5*tau1*Sz_bc)
    c_ops.append(0.5*tau2*Sz_bc_d)

    return c_ops,

print(collapse_ops(1))