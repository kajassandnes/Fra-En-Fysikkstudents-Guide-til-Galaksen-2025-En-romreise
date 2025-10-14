# Ikke brukt kodemal
# Program som har generalisert launch av raketten
from simulere_baner import Planet
import matplotlib.pyplot as plt
import numpy as np

import ast2000tools.utils as utils
import ast2000tools.constants as const
seed = utils.get_seed('oafincke')
from ast2000tools.solar_system import SolarSystem
system = SolarSystem(seed)
from ast2000tools.space_mission import SpaceMission
mission = SpaceMission(seed)

# ========== Konstanter ===========
"""
Vi bruker disse konstantene i oppskytningen:
G (float): gravitasjonskonstanten
M_0 (flaot): masse til planeten vår i kg
R_0 (float): radius til planeten vår i meter
g_0 (float): tyngdeakselerasjonen på planeten vår (ikke konstant)
"""
G = const.G  # m³ / (kgs²)  
M_0 = system.masses[0] * const.m_sun  # kg
R_0 = system.radii[0] * 1e3  # m
g_0 = G*M_0 / R_0**2    # kgm/s²
# =================================

# ============ Rotasjonsfart =============
"""Finner initialhastogheten til raketten fra jordrotasjonen
og unnslipningshastigheten på planetens overflate
"""
rotasjon_dager = system.rotational_periods      # dager / rotasjon
rotasjon_sekunder = utils.day_to_s(rotasjon_dager[0]) # sekunder / rotasjon
vinkelfart = 2*np.pi/rotasjon_sekunder  # radianer / sekund
rotasjonsfart_m_pr_sek = (vinkelfart * R_0)   # m / sekund (kun i yretning)
v_initial_rotation = rotasjonsfart_m_pr_sek   # m/s  
# ============================================

# =========== Relevant til løkka =============
"""Disse parameterne trengs for å launche raketten:
t (float): løpende tid
dt (float): tidssteg
F_motor (float) skyvkraften til motoren
mass_rocket (float): rakettens egenvekt uten drivstoff
mass_initial_fuel (float): mengde drivstoff ved oppskytning
mass_initial (float): total masse ved oppskytning
mass (float): løpende masseendring som tilsvarer rakettens masse etter endt oppskytning
consumption (float): forbruk av drivstoff per sekund
"""
t = 0     # s
dt = 0.01      # sekund 
F_motor = 60000       # N
mass_rocket = 1100      # kg
mass_initial_fuel = 6500    # kg
mass_initial = mass_rocket + mass_initial_fuel  # rakettmasse + drivstoffmasse
mass = mass_initial # kg
consumption = 6.0   # kg/s
# ============================================


def unnslipningsfart(r:np.ndarray[float]):
    """Funksjonen regner ut unnslipningsfarten til planeten basert på rakettens
    avstand fra planetsenteret.

    Vektoriserer radien til planeten og legger den til rakettens avstand til 
    oppskytningspunktet før absoluttverdien av avstanden blir regnet ut. 
    """
    global G, M_0, R_0  # henter variabler utafor funksjonen

    v_escape = np.sqrt(2*G*M_0 / (np.linalg.norm(r + np.array([R_0, 0]))))
    return v_escape


def tyngdeakselerasjon(r):
    """Regner ut tyngdeakselerasjonen Frogstar drar raketten vår nedover
    med basert på rakettens nåværende distanse fra massesenteret.

    Vektoriserer radien til planeten og legger den til rakettens avstand til 
    oppskytningspunktet før absoluttverdien av avstanden blir regnet ut.
    Bruker enhetsvektoren for å få riktig retning på tyngdeakselerasjonen.
    R_hat = R_vec / R 
    """
    global R_0, G, M_0

    R_vec = r + np.array([R_0, 0])
    R = np.linalg.norm(R_vec)
    a = -G*M_0 / R**3 * R_vec
    return a

def planet_position_and_velocity_at_launch(oppskytningstid: float=0, i: int=0) -> np.ndarray:
    """Denne funksjonen gir farts- og posisjonskoordinater etter en viss tid

    Parametere:
    oppskytningstid (float): Tid for launch. Default satt til 0
    i (int): indeks til planet som launchen skjer på. Default satt til planet 0
    """
    # Lager et obkjekt av planet 0
    planet0 = Planet(system.semi_major_axes[i], system.eccentricities[i], \
                system.masses[i], system.radii[i], system.aphelion_angles[i], \
                system.initial_orbital_angles[i], \
                system.initial_positions[0][i], system.initial_positions[1][i], \
                system.initial_velocities[0][i], system.initial_velocities[1][i], i)

    # finner posisjoner og hastigheter for én runde av planeten, samt tidssteg 
    # brukt i simmuleringen
    r_vec, v_vec, dt = planet0.numerisk_bane(runder=1)

    # finner indeksen som tilsvarer planetens posisjon og hastighet ved ønsket
    # oppskytningstid
    indeks = int(oppskytningstid // dt)

    # henter ut startposisjoner og hastigheter
    planet_start_position_vector = r_vec[indeks]    # AU
    planet_start_velocity_vector = v_vec[indeks]    # AU

    return planet_start_position_vector, planet_start_velocity_vector

def launch():
    """Bruker Euler-Cromer til å regne ut ny fart og posisjon til raketten.
    Rakettens akselerasjon finner vi ved å ta rakettens akselerasjon og trekke
    fra tyngdeakselerasjonen. Oppskytningen avsluttes når raketten har 
    oppnådd unnslipningshastighet.

    Resultatet blir at vi får rakettens sluttposisjon og sluttfart i rommet.
  
    Funksjonen lagrer historien til absoluttverdiene til akselerasjonen 
    og farten, slik at de kan plottes mot tid. Posisjonen lagres slik at 
    man kan plotte x-verdier og y-verdier mot hverandre.
    """
    global t, dt, mass, consumption, F_motor

    # ======== Bevegelsesrelevante arrays ==========
    """Arrayen oppdateres med den nyeste posisjonen, farten 
    og akselerasjonen til raketten. Raketten har en initialfart som kommer fra jordrotasjonen."""
    r = np.array([0., 0.])
    v = np.array([0., v_initial_rotation])
    a = np.array([0., 0.])

    """Disse listene lagrer akselerasjon, hastighet og posisjon under løkka, 
    slik at man kan plotte i etterkant."""
    v_skalarer = []
    a_skalarer = []
    r_vektorer = []
    v_vektorer = []
    #a_vektorer = []
    # =============================================

    fortsett = True
    while fortsett:
        # Finner ny unnslipningsfart
        v_escape = unnslipningsfart(r)

        # Euler.Cromer
        a = np.array([F_motor/mass, 0.]) + tyngdeakselerasjon(r)
        v += a * dt
        r += v * dt

        t += dt

        # Lagrer akselerasjon, fart og posisjon, de to øverste er for å plotte
        v_skalarer.append(np.linalg.norm(v)) 
        a_skalarer.append(np.linalg.norm(a))
        r_vektorer.append(np.array([r[0], r[1]]))
        v_vektorer.append(np.array([v[0], v[1]]))
        #a_vektorer.append(np.array([a[0], a[1]]))

               
        # oppdaterer massen
        mass -= consumption * dt

    
        # Dersom vi har oppnådd unnslipningsfart
        if np.linalg.norm(v) >= v_escape:
            fortsett = False


        # Hvis vi bruker opp alt drivstoffet
        if mass < 1100:
            fortsett = False
            return "Du er desverre tom for drivstoff"
    
    r_vektorer = np.array(r_vektorer)
    v_vektorer = np.array(v_vektorer)
    #a_vektorer = np.array(a_vektorer)
        
    return v_skalarer, a_skalarer, r_vektorer, v_vektorer
        
def koordinatskifte(oppskytningstidspunkt:float=0, oppskytningsvinkel:float=0):
    """Gjør koordinatskiftet der brukeren selv velger oppskytningstidspunkt 
    og oppskytningsvinkel.
    
    Argumenter:
    oppskytningstidspunkt (float): tid fra simuleringen starter til oppsytningen (s)
    oppskytningsvinkel (float): vinkel mellom x-aksen i solsystemet og x-aksen 
                                i planetsystemet (der x-aksen i planetsystemet
                                peker radielt med oppskytningen) i radianer.
    """
    v_skalarer, a_skalarer, r_vektorer, v_vektorer = launch()
    planet_position_at_launch, planet_velocity_at_launch = planet_position_and_velocity_at_launch(oppskytningstidspunkt, i=0)

    # ---- Rakettens posisjon i forhold til sola i AU ----
    R_0 = utils.km_to_AU(system.radii[0]) # AU
    rocket_launchpoint = utils.km_to_AU(r_vektorer[-1] * 1e-3)  # AU
    launch_point_moved = planet_velocity_at_launch*utils.s_to_yr(t)
    launch_point_planet = np.array([R_0 * np.cos(oppskytningsvinkel), R_0 * np.sin(oppskytningsvinkel)])  # AU
    planet_star = planet_position_at_launch # AU
    r_rocket_star = rocket_launchpoint + launch_point_moved + launch_point_planet + planet_star # AU

    # ---- Rakettens fart i forhold til sola i AU/year ----
    v_rocket_planet = utils.m_pr_s_to_AU_pr_yr(v_vektorer[-1])  # AU / year
    v_planet_star = planet_velocity_at_launch  #  AU / year
    v_rocket_star = v_rocket_planet + v_planet_star # AU / year

    r_star_launchpoint = planet_star + launch_point_planet # AU

    return v_skalarer, a_skalarer, r_vektorer, v_vektorer, r_rocket_star, v_rocket_star, \
           planet_position_at_launch, planet_velocity_at_launch, r_star_launchpoint




if __name__ == "__main__":
    v_skalarer, a_skalarer, r_vektorer, v_vektorer, r_rocket_star, v_rocket_star, \
    planet_position_at_launch, planet_velocity_at_launch, r_star_launchpoint \
    = koordinatskifte(oppskytningstidspunkt=0, oppskytningsvinkel=np.pi/2)
    # ========== Lager lister til plots ===========
    tider = np.arange(0, t, dt)


    # ---------------------- plots ------------------------
    # Plotter akselerasjon mot tid
    plt.rcParams.update({'font.size':30})
    plt.plot(tider, a_skalarer, color="red", label="Acceleration")
    plt.grid()
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Acceleration [m/s²]")
    plt.title("Rockets acceleration as function of time")
    plt.show()
    # Plotter fart mot tid
    plt.plot(tider, v_skalarer, color="green", label="Velocity")
    plt.grid()
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Velocity [m/s]")
    plt.title("Rockets velocity as function of time")
    plt.show()
    # Plotter y-posisjon mot x-posisjon
    plt.plot(r_vektorer[:,0]/1e3, r_vektorer[:,1]/1e3, color="blue", label="Position")
    plt.grid()
    plt.legend()
    plt.title("Rockets position with launch point at origin")
    plt.xlabel("Position [km]")
    plt.ylabel("Position [km]")
    plt.axis("equal")
    plt.show()

    # --------------- Informasjon om oppskytninga -----------------
    print()
    print("INFORMASJON OM OPPSKYTNING")
    print(f"Rotasjonsfarten til Frogstar world C: {rotasjonsfart_m_pr_sek:.2f} m/s i y-retning")
    print(f"Startfart fra Frogstar World C rotasjonen: {v_initial_rotation:.2f} m/s")
    print(f"Endelig unnslipningsfart: {unnslipningsfart(r_vektorer[-1]):.2f} m/s")
    print(f"Tid til unnslippningsfart: {t:2f} sekunder ({t/60:.2f} minutter)")
    print()

    print(f"Nåværende masse: {mass:.2f} kg")
    print(f"Drivstoff brukt: {(mass_initial - mass):.2f} kg")
    print(f"Gjenværende drivstoff: {(mass - 1100):.2f} kg")
    print()

    print(f"Rakettens fart relativt til oppskytningspunktet: ({v_vektorer[-1][0]:.2f}, {v_vektorer[-1][1]:.2f}) m/s")
    print(f"Rakettens posisjon relativt til oppskytningspunktet: ({r_vektorer[-1][0]:.2f}, {r_vektorer[-1][1]:.2f}) m")
    print()

    # --------------- Initialverdier til planeten ------------------
    print("Startposisjonene til Frogstar World C ved oppskytning i astronomiske enheter:")
    print(f"({planet_position_at_launch[0]:.6f}, {planet_position_at_launch[1]:.6f})")
    print("Startfart til Frogstar World C i astronomiske enheter per år:")
    print(f"({planet_velocity_at_launch[0]:.6f}, {planet_velocity_at_launch[1]:.6f})")
    print()

    # ------------- Rakettens posisjon og fart i forhold til stjernen ------------
    print(f"Avstand fra raketten til Frogstar: ({r_rocket_star[0]:.4f}, {r_rocket_star[1]:.4f}) AU.")
    print(f"Farten til til raketten i forhold til Frogstar: ({v_rocket_star[0]:.4f}, {v_rocket_star[1]:.4f}) AU/year.")
    print()

    # Denne fungerer ikke med annen startposisjon enn ved t=0
    '''# ------------------- Space mission ---------------
    print("------------ SPACE MISSION --------------")
    home_planet_idx = 0 # The home planet always has index 0
    print('My mission starts on planet {:d}, which has a radius of {:g} kilometers.'
      .format(home_planet_idx, mission.system.radii[home_planet_idx]))
    
    print('My spacecraft has a mass of {:g} kg and a cross-sectional area of {:g} m^2.'
      .format(mission.spacecraft_mass, mission.spacecraft_area))
    
    estimated_time = 987    # s
    mission.set_launch_parameters(F_motor, consumption, mass_initial_fuel, \
                                  estimated_time, r_star_launchpoint, 0)

    if not mission.rocket_launched:
        print('I have not launched the rocket yet. Let us do something about that!')
        mission.launch_rocket()

    mission.verify_launch_result(r_rocket_star)'''


