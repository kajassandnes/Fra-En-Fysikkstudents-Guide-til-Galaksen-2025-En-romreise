# This code simulate the gassparticles in an engine
import ast2000tools.constants as const
import numpy as np
import random as rd


# ========== Konstanter ======================================
N = 100           # number off particles in engine
L = 1 * 10**(-6)    # length of box (m)
T = 3 * 10**3       # temperature (K)
k = const.k_B      # Boltzmann-konstanten
x1, x2 = 0, L     # start og slutt
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
mean = 0    
sigma = np.sqrt(k*T/const.m_H2)

def create_random_velocities(n):
    """
    Funksjon som lager en array med N hastighets-vektorer [v_x, v_y, v_z].
    Komponentene blir tilfeldig valgt med en gauss fordeling.
    """
    return np.random.normal(mean, sigma, (n, 3))

hastigheter = create_random_velocities(N)




# Third step: Follow the movements of the gass particles with time
t = 0
dt = 10**(-12)
T = 10**(-9)
v_array = np.zeros(len(hastigheter)*3)

def sjekk(position, velocity):
    if position >= L:
        v_array[np.where(v_array == 0)[0]]
        return (-1) * velocity
    else:
        return velocity

while t < T:
    for i in range(len(hastigheter)):
        hastigheter[i][0] = sjekk(posisjoner[i][0], hastigheter[i][0])
        hastigheter[i][1] = sjekk(posisjoner[i][1], hastigheter[i][1])
        hastigheter[i][2] = sjekk(posisjoner[i][2], hastigheter[i][2])

    posisjoner[i][0] += hastigheter[i][0]*dt
    posisjoner[i][1] += hastigheter[i][1]*dt
    posisjoner[i][2] += hastigheter[i][2]*dt

    t += dt

print(posisjoner)
print(hastigheter)
print(v_array)

