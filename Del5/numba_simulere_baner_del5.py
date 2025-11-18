# Ikke brukt kodemal
# Program som plotter numeriske planetbaner over analytiske planetbaner
import numpy as np
from numba import njit
import matplotlib.pyplot as plt

import ast2000tools.utils as utils
import ast2000tools.constants as const
seed = utils.get_seed('oafincke')

from ast2000tools.solar_system import SolarSystem
system = SolarSystem(seed)
from ast2000tools.space_mission import SpaceMission
mission = SpaceMission(seed)


# ================ Øker størrelse på tekst på plots ====================
plt.rcParams['axes.labelsize'] = 26
plt.rcParams['axes.titlesize'] = 28
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 20


@njit
def akselerasjon(r_vector: np.ndarray, G:float, M:float) -> np.ndarray:
    """Finner akselerasjonen ved Newtons andre lov a = F/m der m er massen
    til sola og F er gravitasjonskraften mellom planeten og stjernen, 
    F = -(GM*r_hat)/r**2. Krafta er negativ fordi den er tiltrekkende og peker
    fra planeten mot stjernen, mens enhetsvektoren peker fra sola mot planeten.
    
    Parametere: 
    r_vec (np.ndarray): current position of planet in x- og y-coordinates (AU)
    M (float): star mass (solar masses)
    G (float): gravitational constant (AU units)
    r (float): absolute value [AU]
    
    Returnerer:
    aks (np.ndarray): acceleration in x- og y-coordinates (AU/year²)
    """
    r = np.sqrt(r_vector[0]**2 + r_vector[1]**2)

    return -G*M / r**3 * r_vector

    
@njit
def numerisk_bane(r0:np.ndarray, v0:np.ndarray, G:float, M:float, P:float, 
                  rounds:float=1) -> np.ndarray | int:
    """Plotter de numeriske banene ved hjelp av Leap-Frog metoden.
    
    Parametere: 
    r0 (ndarray): initial position of planet (AU)
    v0 (ndarray): initial velocity of planet (AU/year)
    G (float): gravitational constant (AU-units)
    M (flaot): mass of star (in solar masses)
    P (float): period of home planet (years)
    rounds (float): rounds of simulation for home planet

    Returnerer:
    r_vec (ndarray): all simulated position x- and y-coordinates of planet
    v_vec (ndarray): all simulated velocity x- and y-coordinates of planet
    dt_p (float): timestep in simulation (years)
    """
    T_tot = P * rounds  # total simulation time
    time_steps_pr_year = 1e6
    total_time_steps = int(T_tot * time_steps_pr_year)
    dt_p = T_tot / total_time_steps

    # lists for storing information about movement
    r_vec = np.zeros((total_time_steps + 1, 2))
    v_vec = np.zeros((total_time_steps + 1, 2))
    a_vec = np.zeros((total_time_steps + 1, 2))
        
    # setting initial values
    r_vec[0] = r0
    v_vec[0] = v0
    a_vec[0] = akselerasjon(r0, G, M)

    # Bruker Leap-Frog
    for i in range(total_time_steps):
        r_vec[i+1] = r_vec[i] + v_vec[i]*dt_p + 0.5*a_vec[i]*dt_p**2         
        a_vec[i+1] = akselerasjon(r_vec[i+1], G, M)
        v_vec[i+1] = v_vec[i] + 0.5*(a_vec[i] + a_vec[i+1])*dt_p

    return r_vec, v_vec, dt_p
    

def planet_bane(i:int, rounds:float=1) -> np.ndarray | float:
    """This function define constants and set initial conditions before calling
    a function that numerically calculate the orbits of the planet.

    Params: 
    i (int): index of planet that we calculate orbits for
    rounds (float): the time of simulation is determined by the number of rounds
                    of planet 0.
    Returns:
    r_vec (ndarray): positions of planet i (AU)
    v_vec (ndarray): velocities at corresponding position (AY/year)
    dt_p (float): timesteps used in planet orbit simulation
    """
    G = const.G_sol
    M = system.star_mass 
    # Perioden til planet 0. Den sørger for at dt_p er lik for alle planeter
    P = np.sqrt(4*np.pi**2 * system.semi_major_axes[0]**3 / (G * (M + system.masses[0])))

    x0, y0 = system.initial_positions[0][i], system.initial_positions[1][i]
    vx0, vy0 = system.initial_velocities[0][i], system.initial_velocities[1][i]
    r0 = np.array([x0, y0])
    v0 = np.array([vx0, vy0])

    r_vec, v_vec, dt_p = numerisk_bane(r0, v0, G, M, P, rounds)    

    return r_vec, v_vec, dt_p 


if __name__ == "__main__":
    # simulating orbits
    r0, v0, dt_p = planet_bane(0, 7)
    r1, v1, dt_p = planet_bane(1, 7)
    r2, v2, dt_p = planet_bane(2, 7)
    r3, v3, dt_p = planet_bane(3, 7)
    r4, v4, dt_p = planet_bane(4, 7)
    r5, v5, dt_p = planet_bane(5, 7)
    r6, v6, dt_p = planet_bane(6, 7)
    r7, v7, dt_p = planet_bane(7, 7)

    # sort the information
    planet_orbits = np.stack((r0, r1, r2, r3, r4, r5, r6, r7))
    planet_velocities = np.stack((v0, v1, v2, v3, v4, v5, v6, v7))
    dt_p = np.array(dt_p)

    # saving information in file
    np.savez("planet_information_7_rounds", planet_orbits=planet_orbits, \
                                             planet_velocities=planet_velocities, \
                                             dt_p=dt_p)
