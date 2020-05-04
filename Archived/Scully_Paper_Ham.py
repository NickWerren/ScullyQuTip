from qutip import *
import matplotlib.pyplot as plt
import numpy as np
from pylab import *
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation

N=3 #Number of Fock states in Hilbert space.

#field operators

a = tensor(destroy(N), qeye(3))

kappa=1.0 # coupling to heat bath
n_th= 0.75 # bath temperature

nc = a.dag() * a
xc = (a +a.dag())

ustate = basis(3,0)
estate = basis(3,1)
gstate = basis(3,2)

# atomic operators
sm = tensor(qeye(N), destroy(3))
suu= tensor(qeye(N), ustate*ustate.dag())
see= tensor(qeye(N), estate*estate.dag())
sgg= tensor(qeye(N), gstate*gstate.dag())
sue= tensor(qeye(N), ustate*estate.dag())
sug= tensor(qeye(N), ustate*gstate.dag())
seg= tensor(qeye(N), estate*gstate.dag())
I = tensor(qeye(N), qeye(3))

#Hamiltonians
H0 = 0*(suu +see +sgg)
HI = xc.dag()*(sue.dag() +sug.dag()) +(sue +sug)*xc
scully_hamiltonian = H0 +HI
#kirton_hamiltonian =

#time 
tf = 100
tsteps = 100
tlist = np.linspace(0, tf, tsteps)

#pumping
lamsteps = 100
lamlim = 1.0
lamarray = np.linspace(0, 0.1, lamsteps)

#bc detuning
om_bc = 0.0 

#photon detuning is Om-nu
Om = 0.0
nu1 = 0
nu2 = nu1
om_ac = 0
om_ab = 0

#ab and ac detunings from cavity
Delta1 = nu1 - om_ac
Delta2 = nu2 - om_ab

#ab and ac cavity couplings
g1 = 1.0
g2 = 1.0

Tx = 1.00/1
taux = 1.00

# atom and the field are initially prepared with one photon
paa0 = tensor(basis(N,0), basis(3,2))
pbb0 = tensor(basis(N,0), basis(3,1))
pcc0 = tensor(basis(N,0), basis(3,0))
psi0 = tensor(basis(N,0), (basis(3,2)+basis(3,1)).unit())

def collapse_ops(i):
    # Build collapse operators
    c_ops = []   
        
    # Photon decay
    c_ops.append(np.sqrt(kappa * (1+n_th)) * a)
#    c_ops.append(np.sqrt(kappa * n_th) * a.dag())
             
    # Atomic decay
    #gam = 0.01 # Atomic decay rate
        
    # ab and ac lifetime and dephasing
    T1 = Tx
    T2 = T1
#    c_ops.append(np.sqrt(T1) * sba) # a->b
#    c_ops.append(np.sqrt(T2) * sca) # a->c

#    c_ops.append(np.sqrt(T1) * sba) # a->b
#    c_ops.append(np.sqrt(T2) * sca) # a->c
#    def col_coeff_AD1(t, args):  # coefficient function
#        return np.exp((-1/T1)*t)
#    def col_coeff_AD2(t, args):  # coefficient function
#        return np.exp((-1/T2)*t)
#    c_ops.append([sba, col_coeff_AD1]) # a->b
#    c_ops.append([sca, col_coeff_AD2]) # a->c
            
    # bc lifetime and dephasing
    tau1= taux
    tau2= tau1
#    c_ops.append(np.sqrt(tau1) * (scc +sbb)) # b->c
#    c_ops.append(np.sqrt(tau2) * (scc -sbb)) # c->b
#    c_ops.append(np.sqrt(tau1) * scb) # b->c
#    c_ops.append(np.sqrt(tau2) * scb.dag()) # c->b
 #   def col_coeff_BC1(t, args):  # coefficient function
 #       return np.exp((-1/tau1)*t)
 #   def col_coeff_BC2(t, args):  # coefficient function
 #       return np.exp((-1/tau2)*t)
 #   c_ops.append([scb, col_coeff_BC1]) # b->c
 #   c_ops.append([scb.dag(), col_coeff_BC2]) # c->b
    
    # Atomic pump
    lam = i # Atomic pumping rate
    c_ops.append(np.sqrt(lam) * sue.dag()) # u->e
    c_ops.append(np.sqrt(lam) * sug.dag()) # u->g
    
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
    solution1 = mesolve(scully_hamiltonian, psi0, tlist, collapse_ops(i), [xc, nc])
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
    solution2 = mesolve(scully_hamiltonian, psi0, tlist, collapse_ops(i), [xc, nc, suu, see, sgg])
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
ax3 = plt.axes(xlim=(0, 0.1), ylim=(0, 1.0))
    
ax3.plot(lamarray, output3, label="<aa>")
#ax3.plot(lamarray, output4, label="<bb>")
#ax3.plot(lamarray, output5, label="<cc>")
ax3.legend()
ax3.set_xlabel('lambda')
ax3.set_ylabel('Populations')
ax3.set_title('Populations in a 3LS');

plt.show()

#plt.figure(4)
#ax4 = plt.axes(xlim=(0, lamlim), ylim=(0, 1.0))
#line1b, = ax1.plot([], [], lw=2)
#line2b, = ax1.plot([], [], lw=2)
#line3b, = ax1.plot([], [], lw=2)
#line4b, = ax1.plot([], [], lw=2)
#line5b, = ax1.plot([], [], lw=2)

# initialization function: plot the background of each frame
def init2():
    line1b.set_data([], [])
    line2b.set_data([], [])
    line3b.set_data([], [])
    line4b.set_data([], [])
    line5b.set_data([], [])
    return line1, line2, line3, line4,

#c_ops = []  
# Photon decay
Gam = 0.01 # Cavity decay rate
#c_ops.append(np.sqrt(Gam) * a)

# ab and ac lifetime and dephasing
T1 = Tx
T2 = T1
#c_ops.append(np.sqrt(T1) * sba) # a->b
#c_ops.append(np.sqrt(T2) * sca) # a->c
   
# bc lifetime and dephasing
tau1= taux
tau2= tau1
#c_ops.append(np.sqrt(tau1) * (scc -sbb)) # b->c
#c_ops.append(np.sqrt(tau2) * (scc -sbb)) # c->b

# Atomic pump
#lam = 1 # Atomic pumping rate
#c_ops.append(np.sqrt(lam) * sba.dag()) # b->a
#c_ops.append(np.sqrt(lam) * sca.dag()) # c->a

#plt.figure(4)

#ssolution = steadystate(scully_hamiltonian, c_ops)
#output1b = expect(xc,  ssolution)
#output2b = expect(nc,  ssolution)
#output3b = expect(saa, ssolution)
#output4b = expect(sbb, ssolution)
#output5b = expect(scc, ssolution)
#plt.axhline(output1b, color='r', lw=1.5)
#plt.axhline(output2b, color='r', lw=1.5)
#plt.ylim([0, 10])
#plt.xlabel('Time', fontsize=14)
#plt.show()