# Ikke brukt kodemal
# Program som har generalisert launch av raketten
import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from linear_interpolation import linear_interpolation
from drivstoff_boost import rocket_properties

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


def initial_rotation_velocity():
    """Finner initialhastogheten til raketten fra jordrotasjonen
    rotasjon_dager (float): planetens omløpstid rundt egen akse (dager/rotasjon)
    rotasjon_sekunder (float): planetens omløpstid rundt egen akse (sekunder/rotasjon)
    vinkelfart (float): planetens rotasjonelle vinkelfart
    """
    R_0 = system.radii[0] * 1e3  # m
    rotasjon_dager = system.rotational_periods[0] # days
    rotasjon_sekunder = utils.day_to_s(rotasjon_dager) # s
    vinkelfart = 2*np.pi/rotasjon_sekunder  # rad / s
    rotasjonsfart_m_pr_sek = np.array([0, vinkelfart * R_0])   # m/s (kun i yretning)
    return rotasjonsfart_m_pr_sek   # m/s  


@njit
def unnslipningsfart(r:np.ndarray, G:float, M_0:float, R_0:float) -> float:
    """Funksjonen regner ut unnslipningsfarten til planeten basert på rakettens
    avstand fra planetsenteret. Vektoriserer radien til planeten og legger den 
    til rakettens avstand til oppskytningspunktet før absoluttverdien av 
    avstanden blir regnet ut. 

    Params:
    r (ndarray): position of spaceship relative to planet (m)
    G (float): gravitational constant (SI-units)
    M_0 (float): mass of planet (kg)
    R_0 (float): radius of planet (m)

    Returns:
    v_escape (float): escape velocity (m/s)
    """
    R = r + np.array([R_0, 0]) 
    v_escape = np.sqrt(2*G*M_0 / (np.sqrt(R[0]**2 + R[1]**2)))

    return v_escape

@njit
def tyngdeakselerasjon(r:np.ndarray, R_0:float, G:float, M_0:float) -> np.ndarray:
    """Regner ut tyngdeakselerasjonen Frogstar drar raketten vår nedover
    med basert på rakettens nåværende distanse fra massesenteret.

    Params:
    r (ndarray): spaceship position (m)
    R_0 (float): radius of planet (m)
    G (float): gravitational constant (SI-units)
    M_0 (float): mass of planet (kg)
    
    Returns:
    a (ndarray): tyngdeakselerasjon i posisjon r relativt til oppskytningspunktet
    """
    R_vec = r + np.array([R_0, 0])
    R = np.sqrt(R_vec[0]**2 + R_vec[1]**2)

    return - G*M_0 / R**3 * R_vec

def planet_position_and_velocity_at_launch(oppskytningstidspunkt:float=0, i:int=0):
    """Denne funksjonen gir farts- og posisjonskoordinater til planeten etter en
    gitt tid som vi bruker til å beregne oppskytningen med.

    Parametere:
    oppskytningstidspunkt (float): Tid for launch. Default satt til 0 (years)
    i (int): indeks til planet som launchen skjer på. Default satt til planet 0
    
    Returns:
    planet_start_position (ndarray): planet position at launch (AU)
    planet_start_velocity (ndarray): planet velocity at launch (AU)
    """
    planet_orbits = np.load('planet_information_10_rounds.npz')['planet_orbits']
    planet_velocities = np.load('planet_information_10_rounds.npz')['planet_velocities']
    dt_p = np.load('planet_information_10_rounds.npz')['dt_p']

    r_planet0 = planet_orbits[0] 
    v_planet0 = planet_velocities[0]

    planet_start_position = linear_interpolation(r_planet0, oppskytningstidspunkt, dt_p) # AU
    planet_start_velocity = linear_interpolation(v_planet0, oppskytningstidspunkt, dt_p) # AU/year

    return planet_start_position, planet_start_velocity


@njit
def launch(v_initial_rotation, G, M_0, R_0, dt=0.01, mass_loss_rate=6.0, thrust=60000, initial_mass=7600):
    """Kode for å launche raketten fra planeten vår. Oppskytningen avsluttes 
    når raketten har oppnådd unnslipningshastighet.

    Params:
    v_initial_rotation (ndarray): initial rotation from planet (m/s)
    G (float): gravitational constant (SI-units)
    M_0 (float): mass of planet (kg)
    R_0 (float): radius of planet (m)
    dt (float): timesteps
    mass_loss_rate (float): loss of fuel per unit of time (kg/s)
    thrust (float): engine power (N)
    intital_mass (float): total mass of spaceship included fuel (kg)

    Returns:
    r (ndarray): position of spaceship after launch (m)
    v (ndarray): velocity of spaceship after launch (m/s)
    t (float): launch duration (s)
    mass (float): total mass of spacesip after launch (kg)
    """
    t = 0
    mass = initial_mass

    # Arrayen oppdateres med den nyeste posisjonen, farten og akselerasjonen 
    # til raketten. Raketten har en initialfart som kommer fra jordrotasjonen.
    r = np.zeros(2)
    v = v_initial_rotation
    a = np.zeros(2)

    fortsett = True
    while fortsett:
        # Finner ny unnslipningsfart
        v_escape = unnslipningsfart(r, G, M_0, R_0)

        # Euler.Cromer
        a[0] = thrust / mass
        a[1] = 0.0
        acc = tyngdeakselerasjon(r, R_0, G, M_0)
        a[0] += acc[0]
        a[1] += acc[1]
        v += a * dt
        r += v * dt

        t += dt
               
        # oppdaterer massen
        mass -= mass_loss_rate * dt

    
        # Dersom vi har oppnådd unnslipningsfart
        if (np.sqrt(v[0]**2 + v[1]**2)) >= v_escape:
            fortsett = False


        # Hvis vi bruker opp alt drivstoffet
        if mass < 1100:
            fortsett = False
            return r, v, t, mass
        
    return r, v ,t, mass
        

def koordinatskifte(oppskytningstidspunkt:float=0, oppskytningsvinkel:float=0, state=False):
    """Gjør koordinatskiftet der brukeren selv velger oppskytningstidspunkt 
    og oppskytningsvinkel.
    
    Parametere:
    oppskytningstidspunkt (float): tid fra simuleringen starter til oppsytningen (s)
    oppskytningsvinkel (float): oppskytningsvinkel mellom x-aksen i solsystemet og x-aksen 
                                i planetsystemet (der x-aksen i planetsystemet peker radielt
                                med oppskytningen) i radianer.
    state (bool): if True -> plot

    Returns:
    r_star_launchpoint (ndarray): position of launchpoint relative to star (AU)
    r_star_rocket (ndarray): position of spaceship directly after launch (AU)
    v_star_rocket (ndarray): velocity of spaceship directly after launch (AU/year)
    launch_duration (float): duration of launch (s)
    """
    v_initial_rotation = initial_rotation_velocity()
    G = const.G  # m³ / (kgs²)  
    M_0 = system.masses[0] * const.m_sun  # kg
    R_0 = system.radii[0] * 10**3   # m
    r, v, duration_of_launch, mass_after_launch = launch(v_initial_rotation, G, M_0, R_0)    # r and v are arrays with shape (2,)
    pp_before_launch, pv_before_launch = planet_position_and_velocity_at_launch(oppskytningstidspunkt)

    # ---- Rakettens posisjon i forhold til sola i AU ----
    R_0 = utils.km_to_AU(system.radii[0]) # AU
    launchpoint_to_rocket = utils.m_to_AU(r)  # AU
    center_to_launchpoint = np.array([R_0, 0.])    # AU
    center_to_rocket = center_to_launchpoint + launchpoint_to_rocket
    r_cr = center_to_rocket
    phi = oppskytningsvinkel
    rotation_center_rocket = np.array([r_cr[0]*np.cos(phi) - r_cr[1]*np.sin(phi), r_cr[0]*np.sin(phi) + r_cr[1]*np.cos(phi)])
    planet_center_moved = pv_before_launch*utils.s_to_yr(duration_of_launch)
    r_star_rocket = pp_before_launch + planet_center_moved + rotation_center_rocket # AU
    # --------------------------------------------------------
    # Roterer hastighetskomponentene slik at de stemmer overens med oppskytningsvinkel
    v[0] = v[0]*np.cos(phi) - v[1]*np.sin(phi)
    v[1] = v[0]*np.sin(phi) + v[1]*np.cos(phi)

    # ---- Rakettens fart i forhold til sola i AU/year ----
    v_rocket_planet = utils.m_pr_s_to_AU_pr_yr(v)  # AU / year
    v_planet_star = pv_before_launch  #  AU / year
    v_star_rocket = v_rocket_planet + v_planet_star # AU / year
    # ------------------------------------------------------
    # coordinates of launchpoint relative to star
    launch_point_planet = np.array([R_0 * np.cos(phi), R_0 * np.sin(phi)])  # AU
    r_star_launchpoint = pp_before_launch + launch_point_planet # AU

    if state:
        print_information(r, v, r_star_rocket, v_star_rocket, pp_before_launch, pv_before_launch, \
                          r_star_launchpoint, duration_of_launch, mass_after_launch)

    return r_star_launchpoint, r_star_rocket, v_star_rocket, duration_of_launch


def print_information(r_after_launch, v_after_launch, r_star_rocket, v_star_rocket, pp_before_launch, \
                      pv_before_launch, r_star_launchpoint, duration_of_launch, mass):

    planet_orbits = np.load('planet_information_10_rounds.npz')['planet_orbits']
    planet0_orbit = planet_orbits[0]
    # ser hvor på banen til planet0 raketten letter fra
    plt.scatter(r_star_launchpoint[0], r_star_launchpoint[1], label="launchposition")
    plt.plot(planet0_orbit[:, 0], planet0_orbit[:, 1], label="planet 0")
    plt.xlabel("x (AU)")
    plt.ylabel("y (AU)")
    plt.title("Launch position")
    plt.axis('equal')
    plt.legend(loc='upper right')
    plt.show()
    # -------------------------------------------------------------
    G = const.G  # m³ / (kgs²)  
    M_0 = system.masses[0] * const.m_sun  # kg
    R_0 = system.radii[0] * 1e3  # m
    # --------------- Informasjon om oppskytninga -----------------
    print()
    print("INFORMASJON OM OPPSKYTNING")
    print(f"Endelig unnslipningsfart: {unnslipningsfart(r_after_launch, G, M_0, R_0):.2f} m/s")
    print(f"Oppskytningen tok: {duration_of_launch:2f} sekunder ({duration_of_launch/60:.2f} minutter)")
    print()
    print(f"Nåværende masse: {mass:.2f} kg")
    print(f"Drivstoff brukt: {(7600 - mass):.2f} kg")
    print(f"Gjenværende drivstoff: {(mass - 1100):.2f} kg")
    print()
    print(f"Rakettens fart relativt til oppskytningspunktet: ({v_after_launch[0]:.2f}, {v_after_launch[1]:.2f}) m/s")
    print(f"Rakettens posisjon relativt til oppskytningspunktet: ({r_after_launch[0]:.2f}, {r_after_launch[1]:.2f}) m")
    print()
    # --------------- Initialverdier til planeten ------------------
    print("Startposisjonene til Frogstar World C ved oppskytning i astronomiske enheter:")
    print(f"({pp_before_launch[0]:.6f}, {pp_before_launch[1]:.6f})")
    print("Startfart til Frogstar World C i astronomiske enheter per år:")
    print(f"({pv_before_launch[0]:.6f}, {pv_before_launch[1]:.6f})")
    print()
    # ------------- Rakettens posisjon og fart i forhold til stjernen ------------
    print(f"Avstand fra raketten til Frogstar: ({r_star_rocket[0]:.4f}, {r_star_rocket[1]:.4f}) AU.")
    print(f"Farten til til raketten i forhold til Frogstar: ({v_star_rocket[0]}, {v_star_rocket[1]}) AU/year.")
    print()



if __name__ == "__main__":
    oppskytningstidspunkt = 2
    oppskytningsvinkel = np.pi
    r_star_launchpoint, r_star_rocket, v_star_rocket, duration_of_launch = koordinatskifte(oppskytningstidspunkt, oppskytningsvinkel, state=True)
    thrust, mass_loss_rate, initial_fuel_mass = rocket_properties()
    
    # ------------------- Space mission ---------------
    print("------------ SPACE MISSION --------------")
    home_planet_idx = 0 # The home planet always has index 0
    print('My mission starts on planet {:d}, which has a radius of {:g} kilometers.'
      .format(home_planet_idx, mission.system.radii[home_planet_idx]))
    
    print('My spacecraft has a mass of {:g} kg and a cross-sectional area of {:g} m^2.'
      .format(mission.spacecraft_mass, mission.spacecraft_area))
    
    estimated_time = duration_of_launch + 1    # s

    mission.set_launch_parameters(thrust, mass_loss_rate, initial_fuel_mass, \
                                  estimated_time, r_star_launchpoint, oppskytningstidspunkt)

    if not mission.rocket_launched:
        print('I have not launched the rocket yet. Let us do something about that!')
        mission.launch_rocket()

    mission.verify_launch_result(r_star_rocket)


