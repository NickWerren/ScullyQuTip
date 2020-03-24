from qutip import *
import matplotlib.pyplot as plt
import numpy as np
from pylab import *
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation

#coherent state 
def CS(N, alpha, offset=0, method='operator'):
    x = basis(N, 0)
    a = destroy(N)
    D = (alpha * a.dag() - conj(alpha) * a).expm()
    return D * x

# The expectation values of the photon number
def Mean_photon(level_n, states):
    a= tensor(destroy(N), qeye(level_n))
    nc = a.dag() * a
    NC1 = expect(nc, states)
    return NC1

def Second_order(level_n, states):
    a= tensor(destroy(N), qeye(level_n))
    ncc= (a.dag()*a)**2
    nc = a.dag() * a
    NC1 = expect(nc, states)
    NCC1 = expect(ncc, states)
    Q=(NCC1-NC1)/NC1**2
    return Q    

def Normal_Sqx(level_n, states):
    a= tensor(destroy(N), qeye(level_n))
    nc = a.dag() * a
    nf = a + a.dag()
    aa = a**2
    ad = a.dag()**2
    aae= expect(aa, states)
    ade= expect(ad, states)
    nfe= expect(nf, states)
    NC1 = expect(nc, states)
    Sx = 0.25*(2* NC1+ aae+ 1+ ade- (nfe)**2)-0.25  
    return Sx

def Normal_Sqy(level_n, states):
    a= tensor(destroy(N), qeye(level_n))
    nc = a.dag() * a
    nf = a - a.dag()
    aa = a**2
    ad = a.dag()**2
    aae= expect(aa, states)
    ade= expect(ad, states)
    nfe= expect(nf, states)
    NC1 = expect(nc, states)
    Sy = (2*NC1- aae- ade +1+ (nfe)**2)*(1/4)-(1/4)    
    return Sy
 
N = 60
g1=1
g2=1    
lam=np.pi/4

# The Hamiltonian of the atom-field system 
def hamiltonian_t(t, args):
    H1 = args[0]
    H2 = args[1]
    H3 = args[2]
    H4 = args[3]
    return a.dag()*((g1-g2*np.exp(1j*Nbeta2))*H1+ (g2+g1*np.exp(-1j*Nbeta2))*H2)+\
           ((g1-g2*np.exp(-1j*Nbeta2))*H3+ (g2+g1*np.exp(1j*Nbeta2))*H4)*a

#field operators
a = tensor(destroy(N), qeye(3))
nc = a.dag() * a
xc = a + a.dag()

# atomic operators
sm = tensor(qeye(N), destroy(3))
scc= tensor(qeye(N), basis(3,0)*basis(3,0).dag())
sbb= tensor(qeye(N), basis(3,1)*basis(3,1).dag())
saa= tensor(qeye(N), basis(3,2)*basis(3,2).dag())
scb= tensor(qeye(N), basis(3,0)*basis(3,1).dag())
sca= tensor(qeye(N), basis(3,0)*basis(3,2).dag())
sba= tensor(qeye(N), basis(3,1)*basis(3,2).dag())
nq = sm.dag() * sm
xq = sm + sm.dag()
sz=scc- sbb
I = tensor(qeye(N), qeye(3))

# dispersive hamiltonian
H1 = sca
H2 = sba
H3 = sca.dag()
H4 = sba.dag()
args = (H1, H2, H3, H4)

# time 
tlist = np.linspace(0, 3.0, 1500)

# collapse operators, only active if gamma1 > 0
c_ops = []

# atom and the field preparation
psiprep =(1/(3**0.5))*(fock(3,0) +fock(3,1) +fock(3,2)) 
psi0 = tensor(CS(N, sqrt(10)), (psiprep).unit())

fig = plt.figure()
ax = plt.axes(xlim=(0, 3), ylim=(0.5, 2.5))
line1, = ax.plot([], [], lw=2)
line2, = ax.plot([], [], lw=2)
line3, = ax.plot([], [], lw=2)
line4, = ax.plot([], [], lw=2)

# initialization function: plots the background of each frame
def init():
    line1.set_data([], [])
    line2.set_data([], [])
    line3.set_data([], [])
    line4.set_data([], [])
    return line1, line2, line3, line4,

# animation function.  This is called sequentially
#def aa_an(i):
#    x = tlist
#    def hamiltonian_t(t, args):
#        H1 = args[0]
#        H2 = args[1]
#        H3 = args[2]
#        H4 = args[3]
#        return a.dag()*((g1- g2*np.exp(1j*i))*H1+ (g2+ g1*np.exp(-1j*i))*H2)+\
#               ((g1- g2*np.exp(-1j*i))*H3+ (g2+ g1*np.exp(1j*i))*H4)*a
#    n2_Iaa = mesolve(hamiltonian_t, psi0, tlist, c_ops, saa, args).expect[0]
#    n2_Ibb = mesolve(hamiltonian_t, psi0, tlist, c_ops, sbb, args).expect[0]
#    line.set_data(x, n2_Iaa)
#    return line,

def maser_anim(i):
    x = tlist
    def hamiltonian_t(t, args):
        H1 = args[0]
        H2 = args[1]
        H3 = args[2]
        H4 = args[3]
        return a.dag()*((g1- g2*np.exp(1j*i))*H1+ (g2+ g1*np.exp(-1j*i))*H2)+\
               ((g1- g2*np.exp(-1j*i))*H3+ (g2+ g1*np.exp(1j*i))*H4)*a
    n2_Iaa = mesolve(hamiltonian_t, psi0, tlist, c_ops, saa, args).expect[0]
    n2_Ibb = mesolve(hamiltonian_t, psi0, tlist, c_ops, sbb, args).expect[0]
    n2_Icc = mesolve(hamiltonian_t, psi0, tlist, c_ops, scc, args).expect[0]
#    n2_In = mesolve(hamiltonian_t, psi0, tlist, c_ops, nq, args).expect[0]
    line1.set_data(x, n2_Iaa)
    line2.set_data(x, n2_Ibb)
    line3.set_data(x, n2_Icc)
#    line4.set_data(x, n2_In)    
    line1.set_label("<aa>")
    line2.set_label("<bb>") 
    line3.set_label("<cc>") 
#    line4.set_label("<cc>") 
    ax.set_xlabel('Time, $t$')
    ax.set_ylabel('P($t$)')
    ax.set_title('Three Level System');
    extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
    plt.legend([extra, line1, line2, line3], (str(round(i,3)), "<aa>", "<bb>", "<cc>", "<n>"))
#    plt.legend(bbox_to_anchor=(0.9, 0.9))
    return line1, line2, line3, #time,

def lin_gain(i):
    x = tlist
    def hamiltonian_t(t, args):
        H1 = args[0]
        H2 = args[1]
        H3 = args[2]
        H4 = args[3]
        return a.dag()*((g1- g2*np.exp(1j*i))*H1+ (g2+ g1*np.exp(-1j*i))*H2)+\
               ((g1- g2*np.exp(-1j*i))*H3+ (g2+ g1*np.exp(1j*i))*H4)*a
    n2_In = mesolve(hamiltonian_t, psi0, tlist, c_ops, nq, args).expect[0]
    line1.set_data(x, n2_In)
    line1.set_label("<n>") 
    ax.set_xlabel('Time, $t$')
    ax.set_ylabel('P($t$)')
    ax.set_title('Gain of a Quantum Beat Laser');
    extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
    plt.legend([extra, line1], (str(round(i,3)), "<n>"))
    return line1, #time,

#anim = FuncAnimation(fig, lin_gain, frames=np.linspace(0, 2*np.pi, 500),init_func=init, blit=True)
#anim.save('maser_animation_pre.gif', fps=10, extra_args=['-vcodec', 'libx264'])

anim = FuncAnimation(fig, lin_gain, frames=np.linspace(0, np.pi, 100),init_func=init, blit=True)
anim.save('maser_animation_gain.gif', fps=10, extra_args=['-vcodec', 'libx264'])


plt.show()