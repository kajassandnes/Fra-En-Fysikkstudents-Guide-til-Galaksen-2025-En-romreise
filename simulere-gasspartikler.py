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
    Funksjon som lager en array med N posisjons-vektorer [x, y, z].
    Komponentene blir tilfeldig valgt med en uniform fordeling innenfor
    'boksen' vår.
    """
    return np.random.uniform(x1, x2, (n, 3))

posisjoner = create_random_positions(N) 




# Second step: Generate random velocities in each directions for all the particles

def create_random_velocities(n):
    """
    Funksjon som lager en array med N hastighets-vektorer [v_x, v_y, v_z].
    Komponentene blir tilfeldig valgt med en gauss fordeling.
    """
    return np.random.normal(mean, sigma, (n, 3))

hastigheter = create_random_velocities(N)



# Third step: Follow the movements of the gass particles with time
# =======================================
t = 0
dt = 1e-12
T = 1e-9
# ========================================

mask = posisjoner[:, 2] < 0
norm = sum(mask)


while t < T:
    kollisjon_topp = posisjoner > L
    kollisjon_bunn = posisjoner < 0
    for i in range(len(posisjoner)):
        kollisjon_bunn[i][2] = False
    

    
    hastigheter[kollisjon_topp | kollisjon_bunn] *= -1

    mask = posisjoner[:, 2] < 0
    norm = sum(mask)
    F = 2*m*norm / dt

    posisjoner += hastigheter*dt

    t += dt

print(posisjoner)
print(hastigheter)
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
