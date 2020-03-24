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
 
g1=1
g2=1    
lam=np.pi/4

# The Hamiltonian of the atom-field system 
# in the presence of classical homogenous gravitational field
def hamiltonian_t(t, args):
    H1 = args[0]
    H2 = args[1]
    H3 = args[2]
    H4 = args[3]
    return a.dag()*((g1-g2*np.exp(1j*Nbeta2))*H1+ (g2+g1*np.exp(-1j*Nbeta2))*H2)+\
           ((g1-g2*np.exp(-1j*Nbeta2))*H3+ (g2+g1*np.exp(1j*Nbeta2))*H4)*a
N = 60

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

# atom and the field are initially prepared in the excited and coherent state
psiprep =(1/(3**0.5))*(fock(3,0) +fock(3,1) +fock(3,2)) 
psi0 = tensor(CS(N, sqrt(10)), (psiprep).unit())


#res = mesolve(hamiltonian_t, psi0, tlist, c_ops, [], args)
#n_Icc = mesolve(hamiltonian_t, psi0, tlist, c_ops, scc, args).expect[0]
#n_Ibb = mesolve(hamiltonian_t, psi0, tlist, c_ops, sbb, args).expect[0]
#n_Iaa = mesolve(hamiltonian_t, psi0, tlist, c_ops, saa, args).expect[0]
#n_In = mesolve(hamiltonian_t, psi0, tlist, c_ops, nq, args).expect[0]


#mean=Mean_photon(3, res.states)
#sec= Second_order(3, res.states)
#Norm=Normal_Sqx(3, res.states)

#plt.figure(1)
#plt.plot(tlist, mean ,"r-")
#plt.xlabel('time')
#plt.ylabel('Mean photon')

#plt.show()
#show()

#plt.figure(2)
#plt.plot(tlist, sec ,"r-")
#plt.xlabel('time')
#plt.ylabel('Second_order')

#plt.show()
#show()

#plt.figure(3)
#plt.plot(tlist, Norm ,"r-")
#plt.xlabel('time (s)')
#plt.ylabel('Normal squeez ')

#plt.show()
#show()

#figure(4)
#fig, ax = plt.subplots(figsize=(8,5))
#ax.plot(tlist, n_e, 'r', label="exponential wavepacket")
#ax.plot(tlist, n_G, 'b', label="Gaussian wavepacket")
#ax.plot(tlist, n_Iaa, 'g', label="<aa>")
#ax.plot(tlist, n_Ibb, 'r', label="<bb>")
#ax.plot(tlist, n_Icc, 'b', label="<cc>")
#ax.plot(tlist, n_Iaa+n_Ibb+n_Icc, 'y', label="total")
#ax.legend()
#ax.set_xlim(0, 3)
#ax.set_ylim(0, 1.25)
#ax.set_xlabel('Time, $t$')
#ax.set_ylabel('Population(t)')
#ax.set_title('Three Level System');

#plt.show()
#show()

#figure(5)
#fig, ax = plt.subplots(figsize=(8,5))
#ax.plot(tlist, n_In, 'b', label="<n>")
#ax.legend()
#ax.set_xlim(0, 3)
#ax.set_ylim(0, 2.0)
#ax.set_xlabel('Time, $t$')
#ax.set_ylabel('Gain(t)')
#ax.set_title('Three Level System');

#plt.show()
#show()

# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
ax = plt.axes(xlim=(0, 3), ylim=(0.5, 2.5))
line1, = ax.plot([], [], lw=2)
line2, = ax.plot([], [], lw=2)
line3, = ax.plot([], [], lw=2)
line4, = ax.plot([], [], lw=2)
#axtext = fig.add_axes([0.0,0.95,0.1,0.05])

# turn the axis labels/spines/ticks off
#axtext.axis("off")

# place the text to the other axes
#time = axtext.text(0.5,0.5, str(0), ha="left", va="top")

# initialization function: plot the background of each frame
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
    
#    line1.set_label("<aa>") # set the label and draw the legend
#    line2.set_label("<bb>") 
#    line3.set_label("<cc>") 
#    time.set_text(str(i))
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
#    line4.set_data(x, n2_In)
    
#    line1.set_label("<aa>") # set the label and draw the legend
#    line2.set_label("<bb>") 
#    line3.set_label("<cc>") 
#    time.set_text(str(i))
    ax.set_xlabel('Time, $t$')
    ax.set_ylabel('P($t$)')
    ax.set_title('Gain of a Quantum Beat Laser');
    extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
    plt.legend([extra, line1], (str(round(i,3)), "<n>"))
#    plt.legend(bbox_to_anchor=(0.9, 0.9))
    return line1, #time,

#anim = FuncAnimation(fig, lin_gain, frames=np.linspace(0, 2*np.pi, 500),init_func=init, blit=True)
#anim.save('maser_animation_pre.gif', fps=10, extra_args=['-vcodec', 'libx264'])

anim = FuncAnimation(fig, lin_gain, frames=np.linspace(0, np.pi, 100),init_func=init, blit=True)
anim.save('maser_animation_gain.gif', fps=10, extra_args=['-vcodec', 'libx264'])
  

#anim.append(FuncAnimation(fig, bb_an, frames=np.linspace(0, 0.5, 10),init_func=init, blit=True))
    
#def cc_an(i):
#    x = tlist
#    def hamiltonian_t(t, args):
#        H1 = args[0]
#        H2 = args[1]
#        H3 = args[2]
#        H4 = args[3]
#        return a.dag()*((g1- g2*np.exp(1j*i))*H1+ (g2+ g1*np.exp(-1j*i))*H2)+\
#               ((g1- g2*np.exp(-1j*i))*H3+ (g2+ g1*np.exp(1j*i))*H4)*a
#    n2_Icc = mesolve(hamiltonian_t, psi0, tlist, c_ops, scc, args).expect[0]
#    line.set_data(x, n2_Icc)
#    return line,

# call the animator.  blit=True means only re-draw the parts that have changed.
#ani = FuncAnimation(fig, aa_an, frames=np.linspace(0, 0.5, 10), init_func=init, blit=True)

#ani.save('basic_animationx.gif', fps=30, extra_args=['-vcodec', 'libx264'])

plt.show()