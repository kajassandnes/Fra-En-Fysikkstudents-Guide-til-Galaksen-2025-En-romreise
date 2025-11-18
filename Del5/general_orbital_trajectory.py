# Ikke brukt kodemal
import numpy as np
import matplotlib.pyplot as plt
from numba import njit

import ast2000tools.utils as utils
import ast2000tools.constants as const
seed = utils.get_seed('oafincke')

from ast2000tools.solar_system import SolarSystem
system = SolarSystem(seed)
from ast2000tools.space_mission import SpaceMission
mission = SpaceMission(seed)

from space_rocket_position_del5 import space_mission
from linear_interpolation import linear_interpolation

# ================ Plotting sizes =================
plt.rcParams['axes.labelsize'] = 26
plt.rcParams['axes.titlesize'] = 28
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 10
# =================================================


def v_stable(r: float) -> float:
    """Function that find the stable circular orbitl velocity of the spaceship
    depending on the distance from the planet
    
    Parameter: 
    r (float): distance from the planet (m)

    Return:
    v (float): stable circular orbital velocity (m/s)
    """
    M = system.masses[1]    # solar_masses
    G = const.G_sol # AU-units

    v = np.sqrt(G*M/r)

    return v


def acceleration(r_ss:np.ndarray, t:float, r_p:np.ndarray, dt_p:float) -> np.ndarray:
    """Calculate accelerasion of spaceship at given time in trajectory, when 
    infuenced by the gravitational force of the sun and all the other planets
    in the solar system.

    Params:
    r_ss (ndarray): spaceship position (AU)
    t (float): time of current position in trajectory (years)
    r_p (plane
    """
    G = const.G_sol # solenheter
    M_sun = system.star_mass

    a = -G * M_sun/np.linalg.norm(r_ss)**3 * r_ss

    for i in range(len(r_p)):
        r_planet = linear_interpolation(r_p[i], t, dt_p)
        a += -G * system.masses[i] / np.linalg.norm(r_ss - r_planet)**3 * (r_ss - r_planet)
    
    return a


def trajectory(t_init:float, r_0:np.ndarray, v_0:np.ndarray, T:float, dt:float):
    """Generalized trajectory for the spaceship depending on initial conditions.
    
    Params:
    t_init (float): initial time of trajectory (year)
    r_0 (ndarray): spaceship initial positions (AU)
    v_0 (ndarray): spaceship initial velocity (AU/year)
    T (float): simulation time (year)
    dt (float): time step for planet (year)

    Return:
    final_time (float): total time of trajectory in years
    final_position (ndarray): final position of trajectory
    final_velocity (ndarray): final velocity of trajectory
    r (ndarray): all spaceship positions through trajectory
    v (ndarray): all spaceship velocities through trajectory
    """
    planet_orbits = np.load('planet_information_7_rounds.npz')['planet_orbits']
    dt_p = np.load('planet_information_7_rounds.npz')['dt_p']

    N = int((T-t_init)/dt)  # antall tidssteg for rakettbanen
    r = np.zeros((N+1, 2))
    v = np.zeros((N+1, 2))
    a = np.zeros((N+1, 2))

    r[0] = r_0
    v[0] = v_0
    a[0] = acceleration(r[0], t_init, planet_orbits, dt_p)

    t = t_init
    # bruker Leap-Frog
    for i in range(N):
        r[i+1] = r[i] + v[i]*dt + 0.5*a[i]*dt**2 
        t += dt        
        a[i+1] = acceleration(r[i+1], t, planet_orbits, dt_p)
        v[i+1] = v[i] + 0.5*(a[i] + a[i+1])*dt 

        

    final_time = t
    final_position = r[-1]
    final_velocity = v[-1]
    
    return final_time, final_position, final_velocity, r, v


def injection_length(r_ss:np.ndarray, r_p:np.ndarray) -> bool:
    """Function that calculate whether spaceship is close enough to do a 
    injection.
    
    Params:
    r_ss (ndarray): position of spaceship (AU)
    r_p (ndarray): position of planet we want to orbit (AU)

    Returns:
    True or False (bool): yes or no
    """
    mass_planet = system.masses[1]
    mass_sun = system.star_mass

    K = np.sqrt(mass_planet / (10*mass_sun))

    r = np.linalg.norm(r_ss)    # distance between sun and spaceship
    l = np.linalg.norm(r_ss - r_p)  # distance between spaceship and planet
    print(f"Vi må ned på {r*K} AU")
    if l < r*K:
        return True
    else:
        return False


def plot_information(r, planet_orbits, final_position, final_velocity, r_0, \
                     start_index, first_index, second_index, third_index, end_index):
    
    if injection_length(r[-1], planet_orbits[1][end_index]):
        print("YESSSSS, YOU CAN DO A INJECTION")

    print(f"AVSTAND ER NÅ: {planet_orbits[1][end_index] - final_position} ---> {np.linalg.norm(planet_orbits[1][end_index] - final_position)}")
    print()
    print(f"FARTE NÅR VI NÅR PLANETEN {utils.AU_pr_yr_to_m_pr_s(final_velocity)} m/s --> {utils.AU_pr_yr_to_m_pr_s(np.linalg.norm(final_velocity))} m/s")
    print(f"STABIL BANEFART: {v_stable(np.linalg.norm(utils.AU_to_m(planet_orbits[1][end_index] - final_position)))} m/s")

    plt.scatter(0, 0, label="sun")
    plt.scatter(r_0[0], r_0[1], label="spaceship start position")
    plt.scatter(r[first_index][0], r[first_index][1], label="spaceship first boost position")
    plt.scatter(r[second_index][0], r[second_index][1], label="spaceship second boost position")
    plt.scatter(r[third_index][0], r[third_index][1], label="spaceship third boost position")
    plt.scatter(final_position[0], final_position[1], marker='x', label="spaceship end position")
    plt.scatter(planet_orbits[0][start_index][0], planet_orbits[0][start_index][1], marker='o', label="planet 0 start position")
    plt.scatter(planet_orbits[1][start_index][0], planet_orbits[1][start_index][1], marker='o', label="planet 1 start position")
    plt.scatter(planet_orbits[0][end_index][0], planet_orbits[0][end_index][1], marker='x', label="planet 0 end position")
    plt.scatter(planet_orbits[1][end_index][0], planet_orbits[1][end_index][1], marker='x', label="planet 1 end position")
    
    plt.plot(r[:, 0], r[:, 1], '--', label="spaceship trajectory")
    plt.plot(planet_orbits[0][start_index:end_index,0], planet_orbits[0][start_index:end_index,1], label="Planet 0")
    plt.plot(planet_orbits[1][start_index:end_index,0], planet_orbits[1][start_index:end_index,1], label="Planet 1")
    plt.xlabel("x [AU]")
    plt.ylabel("y [AU]")
    plt.axis('equal')
    plt.legend()
    plt.show()


def find_planet_launch_positions(planet0:np.ndarray, planet1:np.ndarray, dt_p:float):
    """The function finds the index when the planets are closest to each
    other.

    Params:
    planet0 (ndarray): all positions of planet0 (AU)
    planet1 (ndarray): all positions of planet1 (AU)
    dt (float): timestep used planet orbit simulation (years)

    Returns:
    minimum_index (int): the index gives the position where the distance between
                         planet0 and planet1 is minimal
    time (float): time before planets reach minimal distance (year)
    """
    distance_vec = planet1 - planet0
    distances = np.linalg.norm(distance_vec, axis=1)
    minimum_index = np.argmin(distances)
    # We wish to do the launch one tenth a year before the planets is the closest
    time_min = dt_p * minimum_index
    time = time_min - 0.1
    index_launch = int(time / dt_p)

    return index_launch, time


def start_numerical_trajectory(oppskytningsvinkel:float=0, state=False):
    """Begin the numerical trajectory. Uses three boosts to come to the goal.
    Launch happens 0.1 years before the least distance between the planets 
    occur.
    
    Params: 
    oppskytningsvinkel (int): vinkel relativt til x-aksen (radians)

    Returns:
    planet_position_at_launch (ndarray): (x, y) position of planet at launch (AU)
    time_of_launch (float): when the distance between the planets is smallest (years)
    estimated_launch_duration (float): time to reach escape velocity (years)
    ss_position_after_launch (ndarray): spacecraft position after launch (AU)
    ss_velocity_after_launch (ndarray): spacecraft velocity after launch (AU)
    angle_after_launch (radians): spacecraft angle relative to x-axis after launch (radians)
    final_time (float): end time after trajectory is accomplished (year)
    final_position (ndarray): position of spacecraft after accomplished launch (AU)
    final_velocity (ndarray): velocity of spacecraft after accomplished launch (AU)
    """
    planet_orbits = np.load('planet_information_7_rounds.npz')['planet_orbits']
    dt_p = np.load('planet_information_7_rounds.npz')['dt_p']
    orbit_information_time = dt_p * len(planet_orbits[0])

    # Finner tidspunktet der avstanden mellom planetene er minst 
    start_index, time_of_launch = find_planet_launch_positions(planet_orbits[0], planet_orbits[1], dt_p)
    
    # rakettens posisjon, fart, og tid etter oppskytning
    ss_position_after_launch, ss_velocity_after_launch, duration_of_launch, angle_after_launch, \
    r_star_launchpoint = space_mission(time_of_launch, oppskytningsvinkel)

    # tid gått siden t = 0, til raketten er ferdig med å launche
    t_init = time_of_launch + utils.s_to_yr(duration_of_launch) # tid for å starte å simulere trajectory

    # finner tidssteg for raketten
    time_steps_pr_year = 1e6    # antall tidssteg simulert per år for raketten
    time_steps = int(orbit_information_time * time_steps_pr_year)
    dt_ss = orbit_information_time / time_steps # tidssteg for raketten
    simulation_time = t_init + 0.2
    time_until_first_boost = t_init
    time_until_second_boost = simulation_time - 0.02
    time_until_third_boost = simulation_time - 0.01

    
    # ========================= starting trajectory ===========================
    first_time, first_position, first_velocity, r1, v1 \
    = trajectory(t_init, ss_position_after_launch, ss_velocity_after_launch, \
                 time_until_first_boost, dt_ss)

    # first boost 
    first_boost = np.array([utils.m_pr_s_to_AU_pr_yr(750), utils.m_pr_s_to_AU_pr_yr(-2500)])
    new_velocity = first_velocity + first_boost

    second_time, second_position, second_velocity, r2, v2 \
    = trajectory(first_time, first_position, new_velocity, \
                 time_until_second_boost, dt_ss)  # AU-units

    # second boost
    second_boost = np.array([utils.m_pr_s_to_AU_pr_yr(0), utils.m_pr_s_to_AU_pr_yr(-2000)])
    new_velocity = second_velocity + second_boost

    third_time, third_position, third_velocity, r3, v3 \
    = trajectory(second_time, second_position, new_velocity, \
                 time_until_third_boost, dt_ss)  # AU-units
    
    # third boost
    third_boost = np.array([utils.m_pr_s_to_AU_pr_yr(1660), utils.m_pr_s_to_AU_pr_yr(-1460)])
    new_velocity = third_velocity + third_boost
    
    final_time, final_position, final_velocity, r4, v4 \
    = trajectory(third_time, third_position, new_velocity, \
                 simulation_time, dt_ss)  # AU-units
    
    r = np.vstack((r1, r2, r3, r4))
    v = np.vstack((v1, v2, v3, v4))
    boosts = [(first_boost, time_until_first_boost), (second_boost, time_until_second_boost), (third_boost, time_until_third_boost)]
      

    if state:
        start_index = int(t_init/dt_p)
        first_index = int((first_time-t_init) / dt_ss)
        second_index = int((second_time-t_init) / dt_ss)
        third_index = int((third_time-t_init) / dt_ss)
        end_index = int(final_time / dt_p)
        plot_information(r, planet_orbits, final_position, final_velocity, \
                         ss_position_after_launch, start_index, first_index, \
                         second_index, third_index, end_index)


    estimated_launch_duration = duration_of_launch + 1   # seconds


    return r_star_launchpoint, time_of_launch, estimated_launch_duration, \
           ss_position_after_launch, ss_velocity_after_launch, angle_after_launch,\
           boosts, r, v, dt_ss, final_time



if __name__ == "__main__":
    r_star_launchpoint, time_of_launch, estimated_launch_duration, \
    ss_position_after_launch, ss_velocity_after_launch, angle_after_launch,\
    boosts, r, v, dt_ss, final_time \
    = start_numerical_trajectory(state=True)
    
