from qutip import *
import matplotlib.pyplot as plt
import numpy as np
from pylab import *
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation

# number of Fock states in Hilbert space.
N=3

# field operators
a = tensor(destroy(N), qeye(3))
nc = a.dag() * a
xc = (a +a.dag())

# atom states
astate = basis(3,2)
bstate = basis(3,1)
cstate = basis(3,0)

# atomic operators
saa= tensor(qeye(N), astate*astate.dag())
sbb= tensor(qeye(N), bstate*bstate.dag())
scc= tensor(qeye(N), cstate*cstate.dag())
sab= tensor(qeye(N), astate*bstate.dag())
sac= tensor(qeye(N), astate*cstate.dag())
sbc= tensor(qeye(N), bstate*cstate.dag())
I = tensor(qeye(N), qeye(3))
sm = tensor(qeye(N), destroy(3))

# parameters

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
Delta1 = nu1 - om_ac
Delta2 = nu2 - om_ab

# bc detuning
om_bc = (om_b - om_c)
Delta3 = nu2 - om_bc
#bc_detun = np.exp(1j*((om_bc)*t))

#photon detuning
Om1 = 0.0
Om2 = 0.0
Deltac1 = Om1 - nu1
Deltac2 = Om2 - nu2

# ab and ac cavity couplings
g1 = 1.0
g2 = 1.0

# atom Hamiltonian
H0 = 0.5*(om_a*saa + om_b*sbb + om_c*scc)

# atom-cavity field interaction hamiltonian
Ht1 = g1*xc.dag()*(sab.dag())  # time-dependent term
def Ht1_coeff(t, args):
    ab_detun = np.exp(-1j*((Delta2)*t))
    return ab_detun

Ht2 = g1*xc.dag()*(sac.dag())  # time-dependent term
def Ht2_coeff(t, args):
    ac_detun = np.exp(-1j*((Delta1)*t))
    return ac_detun

Ht3 = g2*(sab)*xc  # time-dependent term
def Ht3_coeff(t, args):
    ab_detun = np.exp(-1j*((Delta2)*t))
    return ab_detun.conjugate()

Ht4 = g2*(sac)*xc  # time-dependent term
def Ht4_coeff(t, args):
    ac_detun = np.exp(-1j*((Delta1)*t))    
    return  ac_detun.conjugate()

# atom-cavity Hamiltonian
phi = 0
field_rot = np.exp(1j*phi)

H_m = (field_rot*sbc.dag() +field_rot.conjugate()*sbc)

scully_hamiltonian = [H0 +H_m, [Ht1,Ht1_coeff],[Ht2,Ht2_coeff],[Ht3,Ht3_coeff],[Ht4,Ht4_coeff]]

# time 
tf = 100
tsteps = 250
tlist = np.linspace(0, tf, tsteps)

# pumping settings
lamsteps = 250
lamlim = 1.0
lamarray = np.linspace(0, 1, lamsteps)

# atom and field initially prepared as a wavefunction
# atom initial wavefunction
psi_at = astate +bstate +cstate
psi0 = tensor(basis(N,0), psi_at).unit()

# atom and field initially prepared as a density matrix
# atom initial density matrix
rho_at_diag = astate*astate.dag() +bstate*bstate.dag() +cstate*cstate.dag()

# atom and field initially prepared with a coherence a la Scully
rho_at_offdiag = cstate*bstate.dag() +bstate*cstate.dag()
rho_at = (rho_at_diag + rho_at_offdiag).unit()

# field initial density matrix
rho_cav  = basis(N,0)*basis(N,0).dag()

# system initial density matrix
rho0 = tensor(rho_cav, rho_at)

def collapse_ops(i):
    # build collapse operators
    c_ops = []   
    
    # incoherent pumping
    lam = i # incoherent pumping rate to a from b and c.
    c_ops.append(np.sqrt(lam) * sab) # b->a
    c_ops.append(np.sqrt(lam) * sac) # c->a
        
    # cavity decay rate 
    Gam=0.01 # coupling to heat bath
    n_th= 1.0 # bath temperature
    c_ops.append(sqrt(Gam * (1+n_th)) * a)
    c_ops.append(sqrt(Gam * n_th) * a.dag())
    
    #photon detuning is Om-nu
    Om = 0.1
    nu = 0.0
    c_ops.append(1j*sqrt( (Om -nu) * (1+n_th)) * a)
    c_ops.append(1j*sqrt( (Om -nu)* n_th) * a.dag())

    # ab and ac lifetime and dephasing
    T1 = 1.0
    T2 = 1.0
    Sz_ac_ab = ((saa-scc)+(saa-sbb))
    Sz_ac_ab_d = ((scc-saa)+(sbb-saa))
    c_ops.append(0.5*T1*Sz_ac_ab)
    c_ops.append(0.5*T2*Sz_ac_ab_d)
    
    # bc lifetime and dephasing
    tau1 = 1.0
    tau2 = 1.0
    Sz_bc = (sbb-scc)
    Sz_bc_d = (scc-sbb)
    c_ops.append(0.5*tau1*Sz_bc)
    c_ops.append(0.5*tau2*Sz_bc_d)

    return c_ops,

plt.figure(1)
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

anim = FuncAnimation(plt.figure(1), Scully_ME, frames=lamarray, init_func=init, blit=True)
anim.save('maser_animation_ScullyE.gif', fps=10, extra_args=['-vcodec', 'libx264'])

plt.show()

plt.figure(2)
ax2 = plt.axes(xlim=(0, lamlim), ylim=(-5, 5))

output1 = []
output2 = []
output3 = []
output4 = []
output5 = []

# Collects expectation values
def Scully_Populations(i):
    solution2 = mesolve(scully_hamiltonian, rho0, tlist, collapse_ops(i), [xc, nc, saa, sbb, scc])
    output1.append(solution2.expect[0][tsteps-1])
    output2.append(solution2.expect[1][tsteps-1])
    output3.append(abs(solution2.expect[2][tsteps-1]))
    output4.append(abs(solution2.expect[3][tsteps-1]))
    output5.append(abs(solution2.expect[4][tsteps-1]))
    ax2.set_xlabel('Time, $t$')
    ax2.set_ylabel('E($t$)')
    ax2.set_title('Electric Field in a 3LS');
    extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
    plt.legend([extra, line1, line2], (str(round(i,3)), "<xc>", "<nc>"))
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
ax3 = plt.axes(xlim=(0, plotlim3), ylim=(0, 1.0))
    
ax3.plot(lamarray, output3, label="<aa>")
ax3.plot(lamarray, output4, label="<bb>")
ax3.plot(lamarray, output5, label="<cc>")
ax3.legend()
ax3.set_xlabel('lambda')
ax3.set_ylabel('Populations')
ax3.set_title('Populations in a 3LS');

plt.show()

plt.figure(4)
plotlim4=0.1
ax3 = plt.axes(xlim=(0, plotlim4), ylim=(0, 1.0))
    
ax3.plot(lamarray, output3, label="<aa>")
ax3.plot(lamarray, output4, label="<bb>")
ax3.plot(lamarray, output5, label="<cc>")
ax3.legend()
ax3.set_xlabel('lambda')
ax3.set_ylabel('Populations')
ax3.set_title('Populations in a 3LS');

plt.show()