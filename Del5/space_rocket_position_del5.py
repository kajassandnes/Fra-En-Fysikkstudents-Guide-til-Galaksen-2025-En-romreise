# Ikke brukt kodemal
import numpy as np
import matplotlib.pyplot as plt

from numba_generalized_launch_del5 import koordinatskifte
from space_rocket_velocity import kartesisk_fart_rakett
from generating_reference_picture import image_analysis
from linear_interpolation import linear_interpolation
from drivstoff_boost import rocket_properties

import ast2000tools.utils as utils
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

def trilateration(r_star_rocket, t:float, distances:np.ndarray, state:bool=False) -> np.ndarray:
    """This is a function that determines our spacecrafts position based on a
    list of distances from the spacecraft to the sun and other planets at 
    specific times.

    Params:
    r_star_rocket (ndarray): numerical position of spaceship relative to star (AU)
                             used in plotting
    t (float): time of measurements (years)
    distance (list): list with measured distances of object from spacecraft (AU)
                     last element in list contains distance to Frogstar
    state (bool): if True -> plot, if False -> do not plot

    Returns:
    position (ndarray): position of spaceship in x- and y-coordinates relative
                        to Frogstar (AU)
    """
    # finding orbit positions in AU
    planet_orbits = np.load('planet_information_7_rounds.npz')['planet_orbits']
    dt_p = np.load('planet_information_7_rounds.npz')['dt_p']

    orbit_one = planet_orbits[1] 
    orbit_six = planet_orbits[6]

    # Finding distances from spaceship
    d_sun = distances[-1] # AU
    d_one = distances[1] # AU
    d_six = distances[6] # AU

    # finding positions to objects at given time
    position_one = linear_interpolation(orbit_one, t, dt_p)
    position_six = linear_interpolation(orbit_six, t, dt_p)

    # Analytical solution
    A = 2 * position_six[1]
    B = 2 * position_one[1]
    C = 2 * position_one[0]
    D = 2 * position_six[0]
    E = d_sun**2 - d_one**2 + position_one[0]**2 + position_one[1]**2
    F = d_sun**2 - d_six**2 + position_six[0]**2 + position_six[1]**2
    G = 4*position_one[0]*position_six[1] - 4*position_six[0]*position_one[1]

    # finding analytical solution
    x = (A*E - B*F) / G     
    y = (C*F - D*E) / G

    position = np.array([x, y])
    
    if state:
        trilateration_plot(r_star_rocket, d_sun, position_one, d_one, position_six, d_six, x, y)

    return position


def trilateration_plot(r_star_rocket, d_sun, position_one, d_one, position_six, d_six, x, y):
    """Vizualizing the trilateration"""
    fig, ax = plt.subplots()
    ax.add_patch(plt.Circle((0,0), d_sun, color="red", fill=False, label="sun"))
    ax.add_patch(plt.Circle(position_one, d_one, color="purple", fill=False, label="Planet 1"))
    ax.add_patch(plt.Circle(position_six, d_six, color="blue", fill=False, label="Planet 6"))
    plt.scatter(x, y, s = 100, color="black", label="analytisk")
    plt.scatter(r_star_rocket[0], r_star_rocket[1], s = 50, color="red", label="numerisk")

    fig.legend()
    ax.set_aspect('equal')
    ax.relim()
    ax.autoscale_view()
    plt.show()


def space_mission(oppskytningstidspunkt=0, oppskytningsvinkel=0):
    thrust, mass_loss_rate, initial_fuel_mass = rocket_properties() # (N, kg/s, kg) 
    r_star_launchpoint, r_star_rocket, v_star_rocket, duarion_of_launch = koordinatskifte(oppskytningstidspunkt, oppskytningsvinkel)
    # ------------------- Space mission ---------------
    print("------------ SPACE MISSION --------------")
    home_planet_idx = 0 # The home planet always has index 0
    print('My mission starts on planet {:d}, which has a radius of {:g} kilometers.'
      .format(home_planet_idx, mission.system.radii[home_planet_idx]))
    
    print('My spacecraft has a mass of {:g} kg and a cross-sectional area of {:g} m^2.'
      .format(mission.spacecraft_mass, mission.spacecraft_area))
    
    estimated_time = duarion_of_launch + 1    # s
    mission.set_launch_parameters(thrust, mass_loss_rate, initial_fuel_mass, \
                                  estimated_time, r_star_launchpoint, oppskytningstidspunkt)

    if not mission.rocket_launched:
        print('I have not launched the rocket yet. Let us do something about that!')
        mission.launch_rocket()

    mission.verify_launch_result(r_star_rocket)
    mission.take_picture('sky_picture.png')
    doppler_shifts = mission.measure_star_doppler_shifts() # nanometers (lambda0, lambda1)
    distances = mission.measure_distances()
    trilateration_position = trilateration(r_star_rocket, utils.s_to_yr(duarion_of_launch) + oppskytningstidspunkt, distances)
    velocity_lambda = kartesisk_fart_rakett(doppler_shifts[0], doppler_shifts[1])
    angle_after_launch = image_analysis('sky_picture.png')
    mission.verify_manual_orientation(trilateration_position, velocity_lambda, angle_after_launch)
    
    return r_star_rocket, v_star_rocket, duarion_of_launch, angle_after_launch, \
           r_star_launchpoint

if __name__ == "__main__":
    oppskytningstidspunkt = 0
    oppskytningsvinkel = 0
    r_star_rocket, v_star_rocket, duarion_of_launch, angle_after_launch, \
    r_star_launchpoint = space_mission(oppskytningstidspunkt, oppskytningsvinkel)
    mission.verify_launch_result(r_star_rocket)
    distances = mission.measure_distances()
    print(f"Oppskytningstidspunkt: {oppskytningstidspunkt} og duarion_of_launch: {duarion_of_launch}")
    tidspunkt = oppskytningstidspunkt + utils.s_to_yr(duarion_of_launch)
    position = trilateration(r_star_rocket, tidspunkt, distances=distances, state=True)