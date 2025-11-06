# Ikke brukt kodemal
import ast2000tools.utils as utils
import ast2000tools.constants as const
seed = utils.get_seed('oafincke')
import numpy as np
import matplotlib.pyplot as plt
from numba_simulere_baner_del5 import planet_bane
from space_rocket_position_del5 import space_mission
from space_rocket_velocity import kartesisk_fart_rakett

from ast2000tools.solar_system import SolarSystem
system = SolarSystem(seed)
from ast2000tools.space_mission import SpaceMission
mission = SpaceMission(seed)


def v_stable(r: float) -> float:
    """Function that find the stable circular orbitl velocity of the spaceship
    depending on the distance from the planet
    
    Parameter: 
    r (float): distance from the planet

    Return:
    v (float): stable circular orbital velocity
    """
    M = system.masses[1]*const.m_sun    # in kg
    G = const.G # SI-enheter

    v = np.sqrt(G*M/r)

    return v

def acceleration(r:np.ndarray, t:float, r_p:np.ndarray, dt:float):
    """Calculate accelerasion of spaceship at given time in trajectory"""
    G = const.G_sol # solenheter
    M_sun = system.star_mass

    a = -G * M_sun/np.linalg.norm(r)**3 * r

    for i in range(2):
        rest = (t % dt) / dt

        indeks_start = int(t / dt)
        indeks_slutt = indeks_start + 1

        add_position = (r_p[i][indeks_slutt] - r_p[i][indeks_start]) * rest
        r_ = r_p[i][indeks_start] + add_position    # AU

        a += -G * system.masses[i] / np.linalg.norm(r - r_)**3 * (r - r_)
    
    return a



def trajectory(runder, t_init:float, r_0:np.ndarray, v_0:np.ndarray, T:float, dt:float):
    """Generalized trajectory for the spaceship depending on initial conditions.
    
    Parameters:
    t_init (float): initial time of launch in years
    r_0 (ndarray): spaceship initial positions after launch in AU
    v_0 (ndarray): spaceship initial velocity after launch in AU/year
    T (float): simulation time in years
    dt (float): time step in years

    Return:
    final_time (float): total time of trajectory in years
    final_position (ndarray): final position of trajectory
    final_velocity (ndarray): final velocity of trajectory
    """
    start_index = int(t_init / dt)
    planet_orbits = []
    planet_velocities = []
    dt_p = 0

    for i in range(8):
        planet_orbit, planet_velocity, dt_p = planet_bane(i, runder)
        planet_orbits.append(planet_orbit)
    
    np.array(planet_orbits)

    index, tid = find_initial_positions(planet_orbits[0], planet_orbits[1], dt_p)

    N = int(T/dt)
    r = np.zeros((N+1, 2))
    v = np.zeros((N+1, 2))
    a = np.zeros((N+1, 2))

    r[0] = r_0
    v[0] = v_0
    a[0] = acceleration(r[0], t_init, planet_orbits, dt_p)

    t = t_init
    # bruker Euler-Cromer
    for i in range(N):
        r[i+1] = r[i] + v[i]*dt + 0.5*a[i]*dt**2         
        a[i+1] = acceleration(r[i+1], t, planet_orbits, dt_p)
        v[i+1] = v[i] + 0.5*(a[i] + a[i+1])*dt 

        t += dt

    final_time = t
    end_index = int(final_time / dt)

    final_position = r[-1]
    final_velocity = v[-1]

    plt.scatter(r_0[0], r_0[1], label="spaceship start position")
    plt.scatter(0, 0, label="sun")
    plt.scatter(r[-1][0], r[-1][1], marker='x', label="spaceship end position")
    plt.scatter(planet_orbits[0][end_index][0], planet_orbits[0][end_index][1], marker='x', label="planet 0 end position")
    plt.scatter(planet_orbits[1][end_index][0], planet_orbits[1][end_index][1], marker='x', label="spaceship end position")
    plt.plot(r[:, 0], r[:, 1], '--', label="spaceship trajectory")
    plt.plot(planet_orbits[0][start_index:end_index,0], planet_orbits[0][start_index:end_index,1], label="Planet 0")
    plt.plot(planet_orbits[1][start_index:end_index,0], planet_orbits[1][start_index:end_index,1], label="Planet 1")
    plt.axis('equal')
    plt.legend()
    plt.show()


    return final_time, final_position, final_velocity

def find_initial_positions(planet0, planet1, dt):
    distance = planet1 - planet0
    distance = np.linalg.norm(distance, axis=1)
    minimum_distance = min(distance)

    minimum_index = np.argmin(distance)

    time = dt * minimum_index

    print(f"MINST AVSTAND INDEX: {minimum_index}")
    print(f"MINST AVSTAND DISTANSE: {minimum_distance} AU")
    print(f"MINST AVSTAND TID: {time} years")

    return minimum_index, time


if __name__ == "__main__":
    oppskytningstidspunkt = 0
    oppskytningsvinkel = 0
    runder = 20
    ss_position, ss_velocity, time_of_launch = space_mission(oppskytningstidspunkt, oppskytningsvinkel, runder)
    dlambda1 = 0
    dlambda2 = 0
    #ss_velocity = utils.m_pr_s_to_AU_pr_yr(kartesisk_fart_rakett(dlambda1, dlambda2))
    T = oppskytningstidspunkt + 0.5
    time_steps_pr_year = 1e5
    time_steps = int(T * time_steps_pr_year)
    dt = T / time_steps

    t = utils.s_to_yr(time_of_launch)

    final_time, final_position, final_velocity = trajectory(runder, t, ss_position, ss_velocity, T, dt)

