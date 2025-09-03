# This code simulate the gassparticles in an engine
import ast2000tools.constants as const
import numpy as np
import random as rd


# ========== Konstanter ======================================
N = 100           # number off particles in engine
L = 1e-6         # length of box (m)
T = 3e3       # temperature (K)
k = const.k_B      # Boltzmann-konstanten
x1, x2 = 0, L     # start og slutt
m = const.m_H2  # masse til hydrogenmolekyl
mean = 0    
sigma = np.sqrt(k*T/m)
# ============================================================


# First step: simulate random positions to the N particles 

rd.seed(100)

def create_random_positions(n):
    """
    Funksjon som lager en array med n posisjons-vektorer [x, y, z].
    Komponentene blir tilfeldig valgt med en uniform fordeling innenfor
    'boksen' vår.
    """
    return np.random.uniform(x1, x2, (n, 3))

def påfyll():
    eps = 1e-12
    x = np.random.uniform(x1, x2)
    y = np.random.uniform(x1, x2)
    posisjon = np.array([x, y, L-eps])
    return posisjon

posisjoner = create_random_positions(N) 




# Second step: Generate random velocities in each directions for all the particles

def create_random_velocities(n):
    """
    Funksjon som lager en array med N hastighets-vektorer [v_x, v_y, v_z].
    Komponentene blir tilfeldig valgt med en gauss fordeling.
    """
    return np.random.normal(mean, sigma, (n, 3))

hastigheter = create_random_velocities(N)

def summerer_fart(hastigheter, partikkel_rømming):
    norm = sum(hastigheter[partikkel_rømming])

    if isinstance(norm, np.ndarray):
        z_norm = abs(norm[2])
        return z_norm
    else:
        return 0

# Third step: Follow the movements of the gass particles with time
# =======================================
t = 0
dt = 1e-12
T = 1e-9
F = 0
# ========================================


while t < T:
    kollisjon_topp = posisjoner > L
    kollisjon_bunn = posisjoner < 0
    for i in range(len(posisjoner)):
        kollisjon_bunn[i][2] = False
    

    partikkel_rømming = posisjoner[:, 2] < 0
    norm = summerer_fart(hastigheter, partikkel_rømming)


    F += 2*m*norm / dt

    hastigheter[kollisjon_topp | kollisjon_bunn] *= -1

    posisjoner += hastigheter*dt

    posisjoner[partikkel_rømming] = påfyll()
    hastigheter[partikkel_rømming] = create_random_velocities(1)

    t += dt

antall = 2.5e13
print(f"Areal til liten boks er {L**2} boksen er {(L*antall)*L} og {L**2 *antall}")
F *= antall
print(F)






# Vi skal sammenligne middel-energi, trykk, og middel-hastighet
# numerisk mot analytisk

def energi():
    pass

def trykk():
    pass

def hastighet():
    pass



# Trykke: Regne ut kreftene
# dp/dt: 
