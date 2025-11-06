# Ikke brukt kodemal
# Program som har generalisert launch av raketten
from numba_simulere_baner_del5 import planet_bane
import matplotlib.pyplot as plt
import numpy as np
import numba
from numba import njit

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


# =========== Egenskaper til raketten =============
"""Disse parameterne trengs for å launche raketten:
dt (float): tidssteg
F_motor (float) skyvkraften til motoren
mass_rocket (float): rakettens egenvekt uten drivstoff
initial_fuel_mass (float): mengde drivstoff ved oppskytning
initial_mass (float): total masse ved oppskytning
consumption (float): forbruk av drivstoff per sekund
"""
dt = 0.01      # sekund 
F_motor = 60000       # N
mass_rocket = 1100      # kg
consumption = 6.0   # kg/s
initial_fuel_mass = 6500    # kg
initial_mass = mass_rocket + initial_fuel_mass  # rakettmasse + drivstoffmasse
# ============================================

@njit
def unnslipningsfart(r:np.ndarray[float], G:float, M_0:float, R_0:float):
    """Funksjonen regner ut unnslipningsfarten til planeten basert på rakettens
    avstand fra planetsenteret.

    Vektoriserer radien til planeten og legger den til rakettens avstand til 
    oppskytningspunktet før absoluttverdien av avstanden blir regnet ut. 
    """
    R = r + np.array([R_0, 0]) 
    v_escape = np.sqrt(2*G*M_0 / (np.sqrt(R[0]**2 + R[1]**2)))

    return v_escape


@njit
def tyngdeakselerasjon(r:np.ndarray, R_0:float, G:float, M_0:float) -> np.ndarray:
    """Regner ut tyngdeakselerasjonen Frogstar drar raketten vår nedover
    med basert på rakettens nåværende distanse fra massesenteret.

    Vektoriserer radien til planeten og legger den til rakettens avstand til 
    oppskytningspunktet før absoluttverdien av avstanden blir regnet ut.
    Bruker enhetsvektoren R_hat = R_vec / R for å få riktig retning på 
    tyngdeakselerasjonen.
    
    Returns:
    a (ndarray): tyngdeakselerasjon i posisjon r relativt til oppskytningspunktet
    """
    R_vec = r + np.array([R_0, 0])
    R = np.sqrt(R_vec[0]**2 + R_vec[1]**2)

    return - G*M_0 / R**3 * R_vec

def planet_position_and_velocity_at_launch(oppskytningstid:float=0, runder=1, i:int=0):
    """Denne funksjonen gir farts- og posisjonskoordinater til planeten etter en
    gitt tid som vi bruker til å beregne oppskytningen med.

    Parametere:
    oppskytningstid (float): Tid for launch. Default satt til 0 (years)
    i (int): indeks til planet som launchen skjer på. Default satt til planet 0
    rest (float): Et tall mellom 0 og 1. Det er den andelen av det siste 
                  tidssteget som ikke ble tatt med når indeksene gjøres om til 
                  heltall (runder alltid ned ved int())
    """
    # finner posisjoner og hastigheter for én runde av planeten, samt tidssteg 
    # brukt i simmuleringen
    r_planet, v_vec, delta_tid = planet_bane(i, runder)

    # -------- interpolering ----------
    # finner indeksen som tilsvarer planetens posisjon og hastighet ved ønsket
    # oppskytningstid
    rest = (oppskytningstid % delta_tid) / delta_tid

    indeks_start = int(oppskytningstid / delta_tid)
    indeks_slutt = indeks_start + 1

    add_position = (r_planet[indeks_slutt] - r_planet[indeks_start]) * rest
    add_velocity = (v_vec[indeks_slutt] - v_vec[indeks_start]) * rest

    # henter ut startposisjoner og hastigheter
    planet_start_position_vector = r_planet[indeks_start] + add_position    # AU
    planet_start_velocity_vector = v_vec[indeks_start] + add_velocity   # AU / year 

    return planet_start_position_vector, planet_start_velocity_vector, r_planet


@njit
def launch(v_initial_rotation, G, M_0, R_0, dt=0.01, consumption=6.0, F_motor=60000, initial_mass=7600):
    """Bruker Euler-Cromer til å regne ut ny fart og posisjon til raketten.
    Rakettens akselerasjon finner vi ved å ta rakettens akselerasjon og trekke
    fra tyngdeakselerasjonen. Oppskytningen avsluttes når raketten har 
    oppnådd unnslipningshastighet.

    Resultatet blir at vi får rakettens sluttposisjon og sluttfart i rommet.
  
    Funksjonen lagrer historien til absoluttverdiene til akselerasjonen 
    og farten, slik at de kan plottes mot tid. Posisjonen lagres slik at 
    man kan plotte x-verdier og y-verdier mot hverandre.
    """
    t = 0
    mass = initial_mass
    # ======== Bevegelsesrelevante arrays ==========
    """Arrayen oppdateres med den nyeste posisjonen, farten 
    og akselerasjonen til raketten. Raketten har en initialfart som kommer fra jordrotasjonen."""
    r = np.zeros(2)
    v = v_initial_rotation
    a = np.zeros(2)
    # =============================================

    fortsett = True
    while fortsett:
        # Finner ny unnslipningsfart
        v_escape = unnslipningsfart(r, G, M_0, R_0)

        # Euler.Cromer
        a[0] = F_motor / mass
        a[1] = 0.0
        acc = tyngdeakselerasjon(r, R_0, G, M_0)
        a[0] += acc[0]
        a[1] += acc[1]
        v += a * dt
        r += v * dt

        t += dt
               
        # oppdaterer massen
        mass -= consumption * dt

    
        # Dersom vi har oppnådd unnslipningsfart
        if (np.sqrt(v[0]**2 + v[1]**2)) >= v_escape:
            fortsett = False


        # Hvis vi bruker opp alt drivstoffet
        if mass < 1100:
            fortsett = False
            return r, v, t, mass
        
    return r, v ,t, mass
        

def koordinatskifte(oppskytningstid:float=0, oppskytningsvinkel:float=0, runder=1):
    """Gjør koordinatskiftet der brukeren selv velger oppskytningstidspunkt 
    og oppskytningsvinkel.
    
    Argumenter:
    oppskytningstidspunkt (float): tid fra simuleringen starter til oppsytningen (s)
    oppskytningsvinkel (float): oppskytningsvinkel mellom x-aksen i solsystemet og x-aksen 
                   i planetsystemet (der x-aksen i planetsystemet peker radielt
                   med oppskytningen) i radianer.
    """
    # ============ Rotasjonsfart fra Frogstar World A =============
    """Finner initialhastogheten til raketten fra jordrotasjonen:
    rotasjon_dager (float): planetens omløpstid rundt egen akse (dager/rotasjon)
    rotasjon_sekunder (float): planetens omløpstid rundt egen akse (sekunder/rotasjon)
    vinkelfart (float): planetens rotasjonelle vinkelfart
    """
    R_0 = system.radii[0] * 1e3  # m
    rotasjon_dager = system.rotational_periods[0]
    rotasjon_sekunder = utils.day_to_s(rotasjon_dager) 
    vinkelfart = 2*np.pi/rotasjon_sekunder  # radianer / sekund
    rotasjonsfart_m_pr_sek = np.array([0, vinkelfart * R_0])   # m / sekund (kun i yretning)
    #rotasjonsfart_m_pr_sek[0] = rotasjonsfart_m_pr_sek[0]*np.cos(oppskytningsvinkel) - rotasjonsfart_m_pr_sek[1]*np.sin(oppskytningsvinkel)
    #rotasjonsfart_m_pr_sek[1] = rotasjonsfart_m_pr_sek[0]*np.sin(oppskytningsvinkel) + rotasjonsfart_m_pr_sek[1]*np.cos(oppskytningsvinkel)
    v_initial_rotation = rotasjonsfart_m_pr_sek   # m/s  
    # ============================================

    G = const.G  # m³ / (kgs²)  
    M_0 = system.masses[0] * const.m_sun  # kg

    r, v, t, mass = launch(v_initial_rotation, G, M_0, R_0)
    pp_at_launch, pv_at_launch, r_planet = planet_position_and_velocity_at_launch(oppskytningstid, runder, i=0)

    # ---- Rakettens posisjon i forhold til sola i AU ----
    R_0 = utils.km_to_AU(system.radii[0]) # AU
    launchpoint_to_rocket = utils.m_to_AU(r)  # AU
    center_to_launchpoint = np.array([R_0, 0.])    # AU
    center_to_rocket = center_to_launchpoint + launchpoint_to_rocket
    r_cr = center_to_rocket
    phi = oppskytningsvinkel
    rotation_launchpoint_center = np.array([r_cr[0]*np.cos(phi) - r_cr[1]*np.sin(phi), r_cr[0]*np.sin(phi) + r_cr[1]*np.cos(phi)])
    launch_point_moved = pv_at_launch*utils.s_to_yr(t)
    r_star_rocket = launch_point_moved + rotation_launchpoint_center + pp_at_launch # AU

    v[0] = v[0]*np.cos(oppskytningsvinkel) - v[1]*np.sin(oppskytningsvinkel)
    v[1] = v[0]*np.sin(oppskytningsvinkel) + v[1]*np.cos(oppskytningsvinkel)

    # ---- Rakettens fart i forhold til sola i AU/year ----
    v_rocket_planet = utils.m_pr_s_to_AU_pr_yr(v)  # AU / year
    v_planet_star = pv_at_launch  #  AU / year
    v_star_rocket = v_rocket_planet + v_planet_star # AU / year


    launch_point_planet = np.array([R_0 * np.cos(phi), R_0 * np.sin(phi)])  # AU
    r_star_launchpoint = pp_at_launch + launch_point_planet # AU

    print(f"FARTA TIL PLANETEN I FORHOLD TIL STJERNA: {utils.AU_pr_yr_to_m_pr_s(v_planet_star)} m/s")
    print(f"FARTA TIL RAKETTEN I FORHOLD TIL PLANETEN: {utils.AU_pr_yr_to_m_pr_s(v_rocket_planet)} m/s")

    return r, v, r_star_rocket, v_star_rocket, pp_at_launch, pv_at_launch, \
           r_star_launchpoint, r_planet, t, mass, 



def print_information(oppskytningstidspunkt=0, oppskytningsvinkel=0, runder=1):
    r_of_launch, v_of_launch, r_star_rocket, v_star_rocket, pp_at_launch, \
    pv_at_launch, r_star_launchpoint, r_planet, time_of_launch, mass \
    = koordinatskifte(oppskytningstidspunkt, oppskytningsvinkel, runder)

    # ser at raketten letter fra et sted på rakettbanen
    plt.scatter(r_star_launchpoint[0], r_star_launchpoint[1])
    plt.plot(r_planet[:, 0], r_planet[:, 1])
    plt.axis('equal')
    plt.show()

    G = const.G  # m³ / (kgs²)  
    M_0 = system.masses[0] * const.m_sun  # kg
    R_0 = system.radii[0] * 1e3  # m
    # --------------- Informasjon om oppskytninga -----------------
    print()
    print("INFORMASJON OM OPPSKYTNING")
    print(f"Endelig unnslipningsfart: {unnslipningsfart(r_of_launch, G, M_0, R_0):.2f} m/s")
    print(f"Tid til unnslippningsfart: {time_of_launch:2f} sekunder ({time_of_launch/60:.2f} minutter)")
    print()

    print(f"Nåværende masse: {mass:.2f} kg")
    print(f"Drivstoff brukt: {(initial_mass - mass):.2f} kg")
    print(f"Gjenværende drivstoff: {(mass - 1100):.2f} kg")
    print()

    print(f"Rakettens fart relativt til oppskytningspunktet: ({v_of_launch[0]:.2f}, {v_of_launch[1]:.2f}) m/s")
    print(f"Rakettens posisjon relativt til oppskytningspunktet: ({r_of_launch[0]:.2f}, {r_of_launch[1]:.2f}) m")
    print()

    # --------------- Initialverdier til planeten ------------------
    print("Startposisjonene til Frogstar World C ved oppskytning i astronomiske enheter:")
    print(f"({pp_at_launch[0]:.6f}, {pp_at_launch[1]:.6f})")
    print("Startfart til Frogstar World C i astronomiske enheter per år:")
    print(f"({pv_at_launch[0]:.6f}, {pv_at_launch[1]:.6f})")
    print()

    # ------------- Rakettens posisjon og fart i forhold til stjernen ------------
    print(f"Avstand fra raketten til Frogstar: ({r_star_rocket[0]:.4f}, {r_star_rocket[1]:.4f}) AU.")
    print(f"Farten til til raketten i forhold til Frogstar: ({v_star_rocket[0]}, {v_star_rocket[1]}) AU/year.")
    print()

    return r_star_launchpoint, r_star_rocket, v_star_rocket, time_of_launch 




if __name__ == "__main__":
    r_star_launchpoint, oppskytningstidspunkt, r_star_rocket, time_of_launch = print_information()
    # ------------------- Space mission ---------------
    print("------------ SPACE MISSION --------------")
    home_planet_idx = 0 # The home planet always has index 0
    print('My mission starts on planet {:d}, which has a radius of {:g} kilometers.'
      .format(home_planet_idx, mission.system.radii[home_planet_idx]))
    
    print('My spacecraft has a mass of {:g} kg and a cross-sectional area of {:g} m^2.'
      .format(mission.spacecraft_mass, mission.spacecraft_area))
    
    estimated_time = time_of_launch + 1    # s
    mission.set_launch_parameters(F_motor, consumption, initial_fuel_mass, \
                                  estimated_time, r_star_launchpoint, oppskytningstidspunkt)

    if not mission.rocket_launched:
        print('I have not launched the rocket yet. Let us do something about that!')
        mission.launch_rocket()

    mission.verify_launch_result(r_star_rocket)


