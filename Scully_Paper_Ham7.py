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

# atom states
astate = basis(N_l,0)
bstate = basis(N_l,1)
cstate = basis(N_l,2)

# define the field

N=3 # number of Fock states in Hilbert space of mode 1.
Two_Mode = False # two modes or one
RWA = True # rotating wave approximation  #WRONG!

# fock states
basis_mode_1 = basis(N,0)

# set up modes for E-field
if Two_Mode is True:
    M = 3
    basis_mode_2 = basis(M,0)
    I_f = tensor(qeye(N), qeye(M))
if Two_Mode is False:
    I_f = qeye(N)

# atomic operators
def ScaleUpOp(x):
    BaseArray = Qobj(np.zeros(shape(x)),dims(x))
    # tensors operator up to the level of N_a atoms
    for i in range(1, N_a):
        BaseArray = tensor(BaseArray, qeye(N_l))
#    print(BaseArray)
    # create an empty subspace of the operator    # now take the operator and tensor it up for each atom
    for i in range(N_a):
        if i == 0:
            SubspaceObj = x
            for j in range(1, N_a):
                    SubspaceObj = tensor(SubspaceObj,qeye(N_l))
#            print(SubspaceObj)
            BaseArray = Qobj(BaseArray.data + SubspaceObj.data)
#            print(BaseArray)
        else:
            SubspaceObj = qeye(N_l)
            for j in range(1, N_a):
                if j == i:
                    SubspaceObj = tensor(SubspaceObj, x)
                else:
                    SubspaceObj = tensor(SubspaceObj, qeye(N_l))
#            print(SubspaceObj)
            BaseArray = Qobj(BaseArray.data + SubspaceObj.data)
#            print(BaseArray)
    return BaseArray    

if RWA is False:
    RWA_switch=1
if RWA is True:
    RWA_switch=0
    
# field operators

if Two_Mode is True:
    # two mode field operators
    a_1 = tensor(destroy(N), qeye(M))    
    b_1 = tensor(qeye(N), destroy(M))  
    a = tensor(a_1 +b_1, (1/N_a)*ScaleUpOp(qeye(N_l)))
if Two_Mode is False:
    # single mode field operator
    a = tensor(destroy(N), (1/N_a)*ScaleUpOp(qeye(N_l)))

xc = (a + RWA_switch*a.dag()) # electric field operator
nc = a.dag() * a # number operator

# atomic operators
Saa = tensor(I_f, ScaleUpOp(astate*astate.dag()))
Sbb = tensor(I_f, ScaleUpOp(bstate*bstate.dag()))
Scc = tensor(I_f, ScaleUpOp(cstate*cstate.dag()))
Sab = tensor(I_f, ScaleUpOp(astate*bstate.dag()))
Sac = tensor(I_f, ScaleUpOp(astate*cstate.dag()))
Sbc = tensor(I_f, ScaleUpOp(bstate*cstate.dag()))
I = tensor(I_f, qeye(N_l))
Sm = tensor(I_f, destroy(N_l))
Sz = tensor(qeye(N), sigmaz())
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

# atom hamiltonian
H_a = ((om_a/2)*Saa + (om_b/2)*Sbb + (om_c/2)*Scc) #JC

# field hamiltonian
H_f = Om1*a.dag()*a

# atom-field interaction hamiltonian
H_af_ac_m = (Om1)*(Sac)  # b -> a, a-dag & sigma minus, time-dependent term
def H_af_ac_m_coeff(t, args):
    ab_detun = np.exp(1j*((om_a)*t))
    return ab_detun

H_af_ac_p = (Om1)*(Sac.dag())  # c -> a, a-dag & sigma minus, time-dependent term
def H_af_ab_m_coeff(t, args):
    ac_detun = np.exp(-1j*((om_a)*t))
    return ac_detun

H_af_ab_m = (Om2)*(Sab)  # a -> b, a & sigma plus, time-dependent term
def H_af_ac_p_coeff(t, args):
    ab_detun = np.exp(1j*((om_c)*t))
    return ab_detun

H_af_ab_p = (Om2)*(Sab.dag())  # a -> c, a & sigma plus, time-dependent term
def H_af_ab_p_coeff(t, args):
    ac_detun = np.exp(-1j*((om_c)*t))    
    return  ac_detun

# atom-field interaction hamiltonian
H_c = g1*(a.dag()*Sbc.dag() + a*Sbc)

# microwave field hamiltonian
phi = 0
field_rot = np.exp(1j*phi)
H_m = (field_rot*Sbc.dag() +field_rot.conjugate()*Sbc)

scully_hamiltonian = [H_a +H_f +H_m +H_c, \
                     [H_af_ac_m,H_af_ac_m_coeff], \
                     [H_af_ab_m,H_af_ab_m_coeff], \
                     [H_af_ac_p,H_af_ac_p_coeff], \
                     [H_af_ab_p,H_af_ab_p_coeff]]

# time 
tf = 300
tsteps = 300
tlist = np.linspace(0, tf, tsteps)

# pumping settings
lamsteps = 250
lamlim = 0.1
lamarray = np.linspace(0, lamlim, lamsteps)

# atom and field initially prepared as a wavefunction
# atom initial wavefunction

def ScaleUpAt(x):
    InitAt = x
    # tensors operator up to the level of N_a atoms
    for i in range(1, N_a):
        InitAt = tensor(InitAt, x)
    return InitAt   

psi_at = astate +bstate +cstate

psi_At = ScaleUpAt(psi_at)

if Two_Mode is True:
    psi0 = tensor(tensor(basis_mode_1,basis_mode_2), psi_At.unit())
if Two_Mode is False:
    psi0 = tensor(basis(N,0), psi_At.unit())

# atom and field initially prepared as a density matrix
# atom initial density matrix
    
rho_at_diag = astate*astate.dag() +bstate*bstate.dag() +cstate*cstate.dag()

# atom and field initially prepared with a coherence a la Scully
rho_at_offdiag = cstate*bstate.dag() + bstate*cstate.dag()
rho_at = (rho_at_diag + rho_at_offdiag)

rho_At = ScaleUpAt((rho_at_diag + rho_at_offdiag))

# field initial density matrix

if Two_Mode is True:
    rho_cav  = tensor(basis_mode_1,basis_mode_2)*tensor(basis_mode_1,basis_mode_2).dag()
    
if Two_Mode is False:
    rho_cav  = basis_mode_1*basis_mode_1.dag()

# system initial density matrix
rho0 = tensor(rho_cav, rho_At.unit())

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
    c_ops.append(np.sqrt(kappa * (1+n_th)) * a) # these change the values of <aa>, <bb>, and <cc> at lam = 0, (<aa> = 0.75, <bb> & <cc> = 0.25)
    c_ops.append(np.sqrt(kappa * n_th) * a.dag())
    
    # atomic relaxation 
    # cavity decay rates
    Gam1=0.01 # b -> a decay rate
    Gam2=0.01 # c -> a decay rate
    c_ops.append(np.sqrt(Gam1) * Sab) # these change the values of <aa>, <bb>, and <cc> at lam = 0, (<aa> = 0.75, <bb> & <cc> = 0.25)
    c_ops.append(np.sqrt(Gam2) * Sac)

    # ab and ac lifetime and dephasing
    T1 = 1.0
    T2 = 1.0
    c_ops.append(0.5*T1*Sz_ac_ab)
    c_ops.append(0.5*T2*Sz_ac_ab_d)
    
    # bc lifetime and dephasing
    tau1 = 1.0
    tau2 = 1.0
    c_ops.append(0.5*tau1*Sz_bc)
    c_ops.append(0.5*tau2*Sz_bc_d)

    return c_ops,

# the animation plot if you are interested in looking into the time dependence

#plt.figure(1)
ax1 = plt.axes(xlim=(0, tf), ylim=(-2.5, 2.5))
line1, = ax1.plot([], [], lw=2)
line2, = ax1.plot([], [], lw=2)

# initialization function: plot the background of each frame
def init():
    line1.set_data([], [])
    line2.set_data([], [])
    return line1, line2,

# animation function.  This is called sequentially
def Scully_ME(i): 
    solution1 = mesolve(scully_hamiltonian, rho0, tlist, collapse_ops(i), [xc, nc])
    line1.set_data(tlist, solution1.expect[0])
#    line2.set_data(tlist, solution1.expect[1])
    line1.set_label("<xc>") # set the label and draw the legend
    line2.set_label("<nc>") 
    ax1.set_xlabel('Time, $t$')
    ax1.set_ylabel('E($t$)')
    ax1.set_title('Electric Field of a Quantum Beat Laser');
    extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
    plt.legend([extra, line1], (str(round(i,3)), "<E>"))
#    plt.legend(bbox_to_anchor=(0.9, 0.9))
    return line1, line2,

#anim = FuncAnimation(plt.figure(1), Scully_ME, frames=lamarray, init_func=init, blit=True)
#anim.save('maser_animation_ScullyE.gif', fps=10, extra_args=['-vcodec', 'libx264'])

#plt.show()

plt.figure(2)
ax2 = plt.axes(xlim=(0, lamlim), ylim=(-(1/2), N+1))
line1b, = ax2.plot([], [], lw=2)
line2b, = ax2.plot([], [], lw=2)

output1 = []
output2 = []
output3 = []
output4 = []
output5 = []

# Collects expectation values
def Scully_Populations(i):
    solution2 = mesolve(scully_hamiltonian, rho0, tlist, collapse_ops(i), [xc, nc, Saa, Sbb, Scc])
    output1.append(solution2.expect[0][tsteps-1])
    output2.append(solution2.expect[1][tsteps-1])
    output3.append(abs(solution2.expect[2][tsteps-1]))
    output4.append(abs(solution2.expect[3][tsteps-1]))
    output5.append(abs(solution2.expect[4][tsteps-1]))
    ax2.set_xlabel('Time, $t$')
    ax2.set_ylabel('E($t$)')
    ax2.set_title('Electric Field in a 3LS');
    extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
    plt.legend([extra, line1b, line2b], (str(round(i,3)), "<xc>", "<nc>"))
#    plt.legend(bbox_to_anchor=(0.9, 0.9))
    return output1, output2, output3, output4, output5,

for i in lamarray:
    Scully_Populations(i)
    
ax2.plot(lamarray, output1, label="<xc>")
ax2.plot(lamarray, output2, label="<nc>")
ax2.legend()
ax2.set_xlabel('lambda')
ax2.set_ylabel('Electric Field')
ax2.set_title('Electric Field in a 3LS');

plt.show()

plt.figure(3)
plotlim3=1.0
ax3 = plt.axes(xlim=(0, lamlim), ylim=(0, N_a))
    
ax3.plot(lamarray, output3, label="<aa>")
ax3.plot(lamarray, output4, label="<bb>")
ax3.plot(lamarray, output5, label="<cc>")
ax3.legend()
ax3.set_xlabel('lambda')
ax3.set_ylabel('Populations')
ax3.set_title('Populations in a 3LS');

plt.show()