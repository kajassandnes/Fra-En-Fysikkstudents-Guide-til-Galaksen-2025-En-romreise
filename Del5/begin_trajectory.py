# Ikke brukt kodemal
import numpy as np
import ast2000tools.utils as utils
import ast2000tools.constants as const
seed = utils.get_seed('oafincke')

from drivstoff_boost import rocket_properties
from general_orbital_trajectory import start_numerical_trajectory, v_stable, injection_length
from linear_interpolation import linear_interpolation

from ast2000tools.solar_system import SolarSystem
system = SolarSystem(seed)
from ast2000tools.space_mission import SpaceMission
mission = SpaceMission(seed)

def v_injection(planet_position, spaceship_position, v_0) -> np.ndarray:
        """Finds the boost needed to do the injection manouver.
        
        Params:
        planet_position (ndarray): position of planet to do the injection
        spaceship_position (ndarray): position of spaceship
        v_0 (ndarray): current velocity of spaceship

        Returns:
        dv_injection (ndarray): change in velocity, boost the spaceship
        """
        r = spaceship_position - planet_position
        distance = np.linalg.norm(r)
        e_r = r / distance
        e_theta = -np.array([-e_r[1], e_r[0]])

        v_stabil = v_stable(distance)
        v = e_theta * v_stabil
        dv_injection = v - v_0

        return dv_injection


def coasting(trajectory, t0, t1, n, r, v, t_init, dt_ss):
    """This function makes small corrections on the spaceship position and 
    velocity, to fit the simulated trajectory.
    
    Params:
    trajectory (InterPlanetaryTravel): object of InterPlanetaryTravel class
    t0 (float): start time of the coasting (yr)
    t1 (float): end ime of the coasting (yr)
    n (int): number of corrections
    r (ndarray): simulated positions of spaceship (AU)
    v (ndarray): simulated velocities of spaceship (AU/yr)
    t_init (float): time when trajectory started (yr)
    dt_ss (float): timestep used for simulation (yr)
    """
    dt = (t1-t0)/n
    for _ in range(n):
        trajectory.coast_until_time(t0+dt)
        tid, posisjon, fart = trajectory.orient()
        #print(f"NUMERICALLY position: {r[int((tid - t_init) / dt_ss)+1]} velocity: {v[int((tid - t_init) / dt_ss)+1]}")
        trajectory.boost(v[int((tid - t_init) / dt_ss)+1] - fart + np.array([utils.m_pr_s_to_AU_pr_yr(0), utils.m_pr_s_to_AU_pr_yr(-153)]))
        #tid, posisjon, fart = trajectory.orient()
        #print(f"NUMERICALLY position: {r[int((tid - t_init) / dt_ss)+1]} velocity: {v[int((tid - t_init) / dt_ss)+1]}")
        print()

        dt += (t1-t0)/n 

    return 


def rocket_trajectory():
    """Function that accomplishes the trajectory."""
    rocket = SpaceMission(seed)
    # ================ brings inn correct initial_values ==================
    pp_at_launch, time_of_launch, estimated_launch_duration, \
    position_after_launch, velocity_after_launch, angle_after_launch,\
    boosts, r, v, dt_ss, final_time = start_numerical_trajectory()
    thrust, mass_loss_rate, initial_fuel_mass = rocket_properties()

    # ================= launching and verifying results ===================
    rocket.set_launch_parameters(thrust, mass_loss_rate, initial_fuel_mass, estimated_launch_duration, pp_at_launch, time_of_launch)
    rocket.launch_rocket()
    rocket.verify_launch_result(position_after_launch)
    rocket.verify_manual_orientation(position_after_launch, velocity_after_launch, angle_after_launch)

    # =============== start rocket trajectory =====================
    t_init = time_of_launch + utils.s_to_yr(estimated_launch_duration)
    planet_orbits = np.load('planet_information_7_rounds.npz')['planet_orbits']
    planet_velocities = np.load('planet_information_7_rounds.npz')['planet_velocities']
    dt_p = np.load('planet_information_7_rounds.npz')['dt_p']

    trajectory = rocket.begin_interplanetary_travel()
    print()

    # ========== FIRST BOOST ============
    tid_f1, posisjon_f1, fart_f1 = trajectory.orient()
    print(f"START: position: {r[0]}, velocity: {v[0]}, tid {t_init}")
    trajectory.boost(v[int((tid_f1 - t_init) / dt_ss)+1] - fart_f1)
    tid_e1, posisjon_e1, fart_e1 = trajectory.orient()
    print(f"NUMERICALLY position: {r[int((tid_e1 - t_init) / dt_ss)+1]} velocity: {v[int((tid_e1 - t_init) / dt_ss)+1]}")
    print()
    coasting(trajectory, t_init, boosts[1][1], 10, r, v, t_init, dt_ss)

    # =========== SECOND BOOST ============
    tid_f2, posisjon_f2, fart_f2 = trajectory.orient()
    print(f"NUMERICALLY position: {r[int((tid_f2 - t_init) / dt_ss)+1]} velocity: {v[int((tid_f2 - t_init) / dt_ss)+1]}")
    trajectory.boost(v[int((tid_f2 - t_init) / dt_ss)+1] - fart_f2 + np.array([utils.m_pr_s_to_AU_pr_yr(664), utils.m_pr_s_to_AU_pr_yr(0)]))
    tid_e2, posisjon_e2, fart_e2 = trajectory.orient()
    print(f"NUMERICALLY position: {r[int((tid_e2 - t_init) / dt_ss)+1]} velocity: {v[int((tid_e2 - t_init) / dt_ss)+1]}")
    print()
    trajectory.coast_until_time(boosts[2][1])

    # ============= THIRD BOOST ==============
    tid_f3, posisjon_f3, fart_f3 = trajectory.orient()
    print(f"NUMERICALLY position: {r[int((tid_f3 - t_init) / dt_ss)+1]} velocity: {v[int((tid_f3 - t_init) / dt_ss)+1]}")
    trajectory.boost(v[int((tid_f3 - t_init) / dt_ss)+1] - fart_f3 + np.array([utils.m_pr_s_to_AU_pr_yr(1000), utils.m_pr_s_to_AU_pr_yr(-1000)]))
    tid_e3, posisjon_e3, fart_e3 = trajectory.orient()
    print(f"NUMERICALLY position: {r[int((tid_e3 - t_init) / dt_ss)+1]} velocity: {v[int((tid_e3 - t_init) / dt_ss)+1]}")
    print()

    # =============== FINAL POSITION ===========
    trajectory.coast_until_time(final_time)
    tid_end, position_end, velocity_end = trajectory.orient()
    print(f"NUMERICALLY position: {r[int((tid_end - t_init) / dt_ss)+1]} velocity: {v[int((tid_end - t_init) / dt_ss)+1]}")
    print()
    injection_length(position_end, linear_interpolation(planet_orbits[1], tid_end, dt_p))
    
    trajectory.take_picture()
    trajectory.look_in_fixed_direction(polar_angle=np.pi)
    trajectory.take_picture(filename="pi_polar_angle_photo.xml")
    trajectory.look_in_fixed_direction(azimuth_angle=np.pi)
    trajectory.take_picture(filename="pi_azimuthal_angle_photo.xml")
    trajectory.look_in_fixed_direction(polar_angle=np.pi, azimuth_angle=np.pi)
    trajectory.take_picture(filename="pi_polar_and_azimuthal_angle_photo.xml")
    trajectory.look_in_direction_of_planet(1)
    trajectory.take_picture(filename='første_av_planet1.xml')

    v_planet = linear_interpolation(planet_velocities[1], tid_end, dt_p)
    v_0 = velocity_end - v_planet
    pp_1 = linear_interpolation(planet_orbits[1], tid_end, dt_p)
    
    dv_injection = v_injection(pp_1, position_end, v_0) 
    trajectory.boost(dv_injection)
    trajectory.coast(utils.s_to_yr(100))
    time, ss_position, ss_velocity = trajectory.orient()

    #trajectory.start_video()

    planet_position = linear_interpolation(planet_orbits[1], time, dt_p)
    planet_velocity = linear_interpolation(planet_velocities[1], time, dt_p)
    distance = np.linalg.norm(ss_position - planet_position)
    distance_to_surface = utils.AU_to_km(distance) - system.radii[1]
    print()
    print(f"Distance to planet center: {utils.AU_to_km(distance)} km.")
    print(f"Distance to planet surface: {distance_to_surface} km.")

    v_rel = ss_velocity - planet_velocity # AU/yr
    r_rel = ss_position - planet_position # AU
    e_r = r_rel / distance
    e_theta = -np.array([-e_r[1], e_r[0]])
    print(f"Distance: {utils.AU_to_m(distance)}")
    print(f"r_rel: {r_rel}")

    theta = np.acos(np.dot(v_rel, e_theta) / np.linalg.norm(v_rel))
    v_theta = utils.AU_pr_yr_to_m_pr_s(np.dot(v_rel, e_theta))
    v_r = utils.AU_pr_yr_to_m_pr_s(np.dot(v_rel, e_r))
    print(f"v_theta: {v_theta} m/s , v_r: {v_r} m/s")
    print()


    trajectory.boost(-utils.m_pr_s_to_AU_pr_yr(v_r) * e_r)
    t, p, v = trajectory.orient()
    distance = np.linalg.norm(p - planet_position)
    v_rel = v - planet_velocity
    r_rel = p - planet_position
    e_r = r_rel / distance
    e_theta = -np.array([-e_r[1], e_r[0]])

    cos_theta = np.dot(v_rel, e_theta) / np.linalg.norm(v_rel)
    cos_theta = np.clip(cos_theta, -1, 1)
    v_theta = utils.AU_pr_yr_to_m_pr_s(np.dot(v_rel, e_theta))
    v_r = utils.AU_pr_yr_to_m_pr_s(np.dot(v_rel, e_r))
    print(f"v_theta: {v_theta*e_theta} m/s , v_r: {v_r*e_r} m/s")
    print()
    orbit_properties(utils.AU_to_m(distance), v_theta, v_r, utils.AU_to_m(r_rel))

    #distance, v_theta, v_r, r_rel = sjekk_bane(trajectory, planet_orbits[1], planet_velocities[1], dt_p)
    #orbit_properties(distance, v_theta, v_r, r_rel)
    #trajectory.coast(0.01)
    #trajectory.finish_video(filename='check_stability_orbit.xml')


def sjekk_bane(trajectory, planet_bane, planet_velocity, dt_p):
    """Let the spaceship coast until the end of the simulated data.
    
    Params:
    trajectory (InterplanetaryTravel): object of InterPlanetaryTravel class
    planet_bane (ndarray): simulated data of planet 1 orbit positions (AU)
    planet_velocity (ndarray): simulated data of planet 1 orbit velocities (AU/yr)
    dt_p (float): timestep used in simulation (yr)

    Returns:
    distance (float): distance between spaceship and planet (m)
    v_theta (ndarray): tangential velocity of spaceship relative to planet (m/s)
    v_r (ndarray): radial velocity of spaceship relative to planet (m/s)
    r (ndarray): x- and y-coordinates of spaceship relative to planet (m)             
    """
    # find total time
    time = (len(planet_bane) - 1) * dt_p

    # coasting
    trajectory.coast_until_time(time)
    time, position, velocity = trajectory.orient()

    # finding new positions values
    r_rel = utils.AU_to_m(position - linear_interpolation(planet_bane, time, dt_p))
    v_rel = utils.AU_pr_yr_to_m_pr_s(velocity - linear_interpolation(planet_velocity, time, dt_p))
    distance = np.linalg.norm(r_rel)
    
    e_r = r_rel / distance
    e_theta = -np.array([-e_r[1], e_r[0]])
    v_theta = utils.AU_pr_yr_to_m_pr_s(np.dot(v_rel, e_theta))
    v_r = utils.AU_pr_yr_to_m_pr_s(np.dot(v_rel, e_r))
    print()
    
    return distance, v_theta, v_r, r_rel


def orbit_properties(distance, v_theta, v_r, r):
    """Alt her bruker SI-enheter.
    
    Params: 
    distance (float): distance between spaceship and planet (m)
    v_theta (ndarray): tangential velocity of spaceship relative to planet (m/s)
    v_r (ndarray): radial velocity of spaceship relative to planet (m/s)
    r (ndarray): x- and y-coordinates of spaceship relative to planet (m)

    Calculations:
    a (float): semi-major axis (m)
    b (float): semi-minor axis (m)
    e (float): eccentricity 
    P (float): time for one revolution around the planet (hours)
    r_min (float): periapsis (m)
    r_max (float): apoapsis (m)
    """
    print()
    m = const.G * (1100 + system.masses[1]*const.m_sun)
    v = np.sqrt(v_theta**2 + v_r**2)
    
    a = 1 / (2/distance - (v)**2/m)
    print(f"a: {a} m")
    
    e = 1/a * (r[0] - np.sqrt(distance**2 - r[1]**2))
    print(f"e: {e}")

    b = a * np.sqrt(1-e**2)
    print(f"b: {b}")

    # Newtons version of Keplers 3rd law:
    P = np.sqrt(4*np.pi*a**3 / m)
    P = utils.s_to_day(P) * 24
    print(f"P: {P} hours")

    # using formula for general solution for the two body problem
    p = a*(e**2 - 1)
    r_min = p / (1 + e)
    r_max = p / (1 - e)
    print(f"periapsis: {r_min*1e-3} km")
    print(f"apoapsis: {r_max*1e-3} km")
    print()
    
         

if __name__ == "__main__":
    rocket_trajectory()
