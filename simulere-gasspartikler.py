# This code simulate the gassparticles in an engine
import ast2000tools.constants as at
import numpy as np
import random as rd

N = 100         # number off particles in engine
L = 1 * 10**(-6)  # length of box (m)
T = 3 * 10**3   # temperature (K)
k = at.k_B
x1, x2 = 0, L

# First step: simulate random positions to the N particles 
rd.seed(100)
number_position = rd.uniform(x1, x2)
posisjoner = np.full((100, 3), number_position)

# Second step: Generate random velocities in each directions for all the particles
mean = k*T
sigma = 0
number_velocity = rd.gauss(mean, sigma)

# Third step: Follow the movements of the gass particles with time
t = 0
dt = 10**(-12)
T = 10**(-9)
while t < T:
    pass

print(posisjoner)