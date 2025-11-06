# Ikke brukt kodemal
import numpy as np
import matplotlib.pyplot as plt
from numba_simulere_baner_del5 import planet_bane
from numba_generalized_launch_del5 import print_information

import ast2000tools.utils as utils
import ast2000tools.constants as const
seed = utils.get_seed('oafincke')
from ast2000tools.solar_system import SolarSystem
system = SolarSystem(seed)
from ast2000tools.space_mission import SpaceMission
mission = SpaceMission(seed)

plt.rcParams['axes.labelsize'] = 26
plt.rcParams['axes.titlesize'] = 28
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 20

def trilateration(r_star_rocket, t:float, distances:np.ndarray) -> np.ndarray:
    """This is a function that determines our spacecrafts position based on a
    list of distances from the spacecraft to the sun and other planets at 
    specific times.

    Parameters:
    t (float): time of measurements
    distance (list): list with measured distances of object from spacecraft (AU)
                     last element in list contains distance to Frogstar 

    
    Return:
    position (ndarray): position of spaceship in x- and y-coordinates relative
                        to Frogstar 
    """
    # finding orbit positions in AU
    orbit_one, v_vec, dt_one = planet_bane(1)  
    orbit_six, v_vec, dt_six = planet_bane(6)

    # Finding distances from spaceship
    d_sun = distances[-1] # AU
    d_one = distances[1] # AU
    d_six = distances[6] # AU

    # finding positions to objects at given time
    indeks_start = int(t / dt_one)
    indeks_slutt = indeks_start + 1
    korreksjon = (t % dt_one) / dt_one

    difference_one = orbit_one[indeks_slutt] - orbit_one[indeks_start]
    add_position_one = difference_one * korreksjon

    difference_six = orbit_six[indeks_slutt] - orbit_six[indeks_start]
    add_position_six = difference_six * korreksjon

    p_one = orbit_one[indeks_start] + add_position_one
    p_six = orbit_six[indeks_start] + add_position_six

    # making circle plots
    fig, ax = plt.subplots()

    ax.add_patch(plt.Circle((0,0), d_sun, color="red", fill=False, label="sun"))
    ax.add_patch(plt.Circle(p_one, d_one, color="purple", fill=False, label="Planet 1"))
    ax.add_patch(plt.Circle(p_six, d_six, color="blue", fill=False, label="Planet 6"))

    # Analytical solution
    A = 2 * p_six[1]
    B = 2 * p_one[1]
    C = 2 * p_one[0]
    D = 2 * p_six[0]

    E = d_sun**2 - d_one**2 + p_one[0]**2 + p_one[1]**2
    F = d_sun**2 - d_six**2 + p_six[0]**2 + p_six[1]**2

    G = 4*p_one[0]*p_six[1] - 4*p_six[0]*p_one[1]

    # plotting analytical solution
    x = (A*E - B*F) / G     
    y = (C*F - D*E)/ G
    plt.scatter(x, y, s = 100, color="black", label="analytisk")
    plt.scatter(r_star_rocket[0], r_star_rocket[1], s = 50, color="red", label="numerisk")
    fig.legend()

    ax.set_aspect('equal')
    ax.relim()
    ax.autoscale_view()

    plt.show()

    position = np.array([x, y])
    
    return position


def space_mission(oppskytningstidspunkt, oppskytningsvinkel, runder=1):
    F_motor = 60000       # N
    mass_initial_fuel = 6500    # kg
    consumption = 6.0   # kg/ssix
    #oppskytningstidspunkt = 0.0
    #oppskytningsvinkel = 0.0
    r_star_launchpoint, r_star_rocket, v_star_rocket, time_of_launch = print_information(oppskytningstidspunkt, oppskytningsvinkel, runder)
    # ------------------- Space mission ---------------
    print("------------ SPACE MISSION --------------")
    home_planet_idx = 0 # The home planet always has index 0
    print('My mission starts on planet {:d}, which has a radius of {:g} kilometers.'
      .format(home_planet_idx, mission.system.radii[home_planet_idx]))
    
    print('My spacecraft has a mass of {:g} kg and a cross-sectional area of {:g} m^2.'
      .format(mission.spacecraft_mass, mission.spacecraft_area))
    
    estimated_time = time_of_launch + 1    # s
    mission.set_launch_parameters(F_motor, consumption, mass_initial_fuel, \
                                  estimated_time, r_star_launchpoint, oppskytningstidspunkt)

    if not mission.rocket_launched:
        print('I have not launched the rocket yet. Let us do something about that!')
        mission.launch_rocket()

    #mission.verify_launch_result(r_star_rocket)

    return r_star_rocket, v_star_rocket, time_of_launch

if __name__ == "__main__":
    r_star_rocket, oppskytningstidspunkt, time_of_launch = space_mission()
    mission.verify_launch_result(r_star_rocket)
    distances = mission.measure_distances()
    print(f"Oppskytningstidspunkt: {oppskytningstidspunkt} og time_of_launch: {time_of_launch}")
    tidspunkt = oppskytningstidspunkt + utils.s_to_yr(time_of_launch)
    trilateration(r_star_rocket, tidspunkt, distances=distances)