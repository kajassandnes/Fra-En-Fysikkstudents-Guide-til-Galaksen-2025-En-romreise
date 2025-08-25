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
posisjoner = np.zeros((100, 3))

def create_random_positions(array):
    """
    Funksjon som lager en array med N posisjons-vektorer [x, y, z].
    Komponentene blir tilfeldig valgt med en uniform fordeling innenfor
    'boksen' vår.
    """
    for i in range(len(array)):
        number_position = rd.uniform(x1, x2)
        array[i] = number_position
    return array
    
posisjoner = create_random_positions(posisjoner)






# Second step: Generate random velocities in each directions for all the particles
mean = 0    
sigma = np.sqrt(k*T/const.m_H2)
hastigheter = np.zeros((100, 3))

def create_random_velocities(array):
    """
    Funksjon som lager en array med N hastighets-vektorer [v_x, v_y, v_z].
    Komponentene blir tilfeldig valgt med en gauss fordeling.
    """
    for i in range(len(array)):
        number_velocity = np.random.normal(mean, sigma)
        array[i] = number_velocity
    return array

hastigheter = create_random_velocities(hastigheter)





# Third step: Follow the movements of the gass particles with time
t = 0
dt = 10**(-12)
T = 10**(-9)
while t < T:
    t += dt

print(posisjoner)
print(hastigheter)