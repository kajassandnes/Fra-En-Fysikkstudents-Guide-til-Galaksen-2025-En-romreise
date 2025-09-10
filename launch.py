# Ikke brukt kodemal, kommentarer til koden i slutten av programmet
# Program som launcher raketten
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
M_0 = system.masses[0] * const.m_sun   # masse til planeten vår i kg
G = const.G    # gravitasjonskonstanten
R_0 = system.radii[0] * 1e3   # radius til planeten vår i meter
g_0 = G*M_0 / R_0**2    # tyngdeakselerasjonen på planeten vår
# =================================

# ============ Rotasjonsfart og unnslipningsfart =============
rotasjon_dager = system.rotational_periods      # dager / rotasjon
rotasjon_sekunder = utils.day_to_s(rotasjon_dager[0]) # sekunder / rotasjon
vinkelfart = 2*np.pi/rotasjon_sekunder  # radianer / sekund
rotasjonsfart_m_pr_sek = (vinkelfart * R_0)   # m / sekund (kun i yretning)

v_escape = np.sqrt(2*G*M_0/R_0)     # m/s
v_initial = rotasjonsfart_m_pr_sek   # m/s  
# ============================================

# ======== Bevegelsesrelevante arrays ==========
r = np.array([0., 0.])
v = np.array([0., v_initial])
a = np.array([0., 0.])
# =============================================

# =========== Relevant til løkka =============
t = 0     # s
dt = 0.01      # sekund 
F_motor = 60000       # N
mass_rocket = 1100      # kg
mass_initial_fuel = 6500    # kg
mass_initial = mass_rocket + mass_initial_fuel  # rakettmasse + drivstoffmasse
mass = mass_initial # kg
consumption = 6.0   # kg/s

# Følgende lister lagrer verdier til plot
v_historie = []
a_historie = []
r_historie = []
# ============================================

def unnslipningsfart(r:np.ndarray[float]):
    """Funksjonen regner ut unnslipningsfarten til planeten basert på rakettens
    avstand fra planetsenteret.
    """
    global G, M_0, R_0  # henter variabler utafor funksjonen

    v_escape = np.sqrt(2*G*M_0 / (R_0 + np.linalg.norm(r)))
    return v_escape


def tyngdeakselerasjon(r):
    """Regner ut tyngdeakselerasjonen Frogstar drar raketten vår nedover
    med basert på rakettens nåværende distanse fra massesenteret.
    """
    global R_0, G, M_0

    R = R_0 + np.linalg.norm(r)
    a = G*M_0 / R**2
    return a

def launch():
    """Bruker Euler-Cromer til å regne ut ny fart og posisjon til raketten.
    Rakettens akselerasjon finner vi ved å ta rakettens akselerasjon og trekke
    fra tyngdeakselerasjonen. Oppskytningen avsluttes når raketten har 
    oppnådd unnslipningshastighet.
  
    Funksjonen lagrer historien til absoluttverdiene til akselerasjonen 
    og farten, slik at de kan plottes mot tid. Posisjonen lagres slik at 
    man kan plotte x-verdier og y-verdier mot hverandre.
    """
    global r, v, a, v_escape, t, dt, mass, consumption, F_motor

    fortsett = True
    while fortsett:
        # Euler.Cromer
        a[0] = F_motor/mass - tyngdeakselerasjon(r)
        v += a * dt
        r += v * dt

        t += dt

        # Lagrer akselerasjon, fart og posisjon
        a_historie.append(np.linalg.norm(a))
        v_historie.append(np.linalg.norm(v)) 
        r_historie.append(np.array([r[0], r[1]]))

               
        # oppdaterer massen
        mass -= consumption * dt

        # Finner ny unnslipningsfart
        v_escape = unnslipningsfart(r)


        # Dersom vi har oppnådd unnslipningsfart
        if np.linalg.norm(v) >= v_escape:
            fortsett = False

        # Hvis vi bruker opp alt drivstoffet
        if mass < 1100:
            fortsett = False
            return "Du er desverre tom for drivstoff"


if __name__ == "__main__":
    launch()
    # ========== Lager lister til plots ===========
    tid_til_unnslipningsfart = np.arange(0, t, dt)
    r_historie = np.array(r_historie)

    # =========== Forenkler variabelnavn ============
    initial_positions = system.initial_positions
    initial_velocities = system.initial_velocities


    # ------------ Rakketens fart i forhold til sola i km/t ---------
    v_rakett_jord = v * 1e-3 # km/s 
    v_planet_sol_x = utils.AU_pr_yr_to_m_pr_s(initial_velocities[0][0]) / 1000  # km/s
    v_planet_sol_y = utils.AU_pr_yr_to_m_pr_s(initial_velocities[1][0]) / 1000   # km/s
    v_planet_sol_vec = np.array([v_planet_sol_x, v_planet_sol_y])      # km / s
    fart_rakett_sol = v_planet_sol_vec + v_rakett_jord  # km/s


    # ---------- Rakketens posisjon i forhold til sola i km ----------------
    planetradien = np.array([system.radii[0], 0])      # km
    avstand_sol_planet_x = utils.AU_to_km(initial_positions[0][0])  # km
    avstand_sol_planet_y = utils.AU_to_km(initial_positions[1][0])  # km
    avstand_sol_planet_vec = np.array([avstand_sol_planet_x, avstand_sol_planet_y])  # km
    koordinater_rakett_oppskytningspunkt = r * 1e-3  # km
    koordinater_rakett_sol = avstand_sol_planet_vec + planetradien + koordinater_rakett_oppskytningspunkt   # km


    # ---------- Rakketens posisjon i forhold til sola i AU ----------------
    planetradie_AU = utils.km_to_AU(system.radii[0])
    avstand_sol_planet_x_AU = initial_positions[0][0]
    avstand_sol_planet_y_AU = initial_positions[1][0]
    avstand_sol_oppskytningspunkt_AU = np.array([avstand_sol_planet_x_AU + planetradie_AU, avstand_sol_planet_y_AU])
    dr_oppskytningspunkt_AU = np.array([initial_velocities[0][0]*utils.s_to_yr(t), initial_velocities[1][0]*utils.s_to_yr(t)])
    koordinater_rakett_oppskytningspunkt_AU = utils.km_to_AU(koordinater_rakett_oppskytningspunkt)
    koordinater_rakett_sol_AU = avstand_sol_oppskytningspunkt_AU + dr_oppskytningspunkt_AU + koordinater_rakett_oppskytningspunkt_AU  # AU



    # ---------------------- plots ------------------------
    # Plotter akselerasjon mot tid
    plt.rcParams.update({'font.size':30})
    plt.plot(tid_til_unnslipningsfart, a_historie, color="red", label="Akselerasjon")
    plt.grid()
    plt.legend()
    plt.xlabel("Tid [s]")
    plt.ylabel("Akselerasjon [m/s²]")
    plt.title("Rakettens akselerasjon som funksjon av tid")
    plt.show()
    # Plotter fart mot tid
    plt.plot(tid_til_unnslipningsfart, v_historie, color="green", label="Fart")
    plt.grid()
    plt.legend()
    plt.xlabel("Tid [s]")
    plt.ylabel("Fart [m/s]")
    plt.title("Rakettens fart som funksjon av tid")
    plt.show()
    # Plotter y-posisjon mot x-posisjon
    plt.plot(r_historie[:,0]/1e3, r_historie[:,1]/1e3, color="blue", label="Posisjon")
    plt.grid()
    plt.legend()
    plt.title("Rakettens posisjon med oppskytningspunkt i origo")
    plt.xlabel("Posisjon [km]")
    plt.ylabel("Posisjon [km]")
    plt.axis("equal")
    plt.show()


    # --------- printer informasjon -----------
    print("GENERELL INFORMASJON")
    print(f"Rotasjonsfarten til Frogstar: {rotasjonsfart_m_pr_sek} m/s i y-retning")
    print(f"Endelig unnslipningsfart: {v_escape} m/s")
    print(f"Startfart fra jordrotasjon: {v_initial} m/s")
    print()

    # --------------- Informasjon om oppskytninga -----------------
    print("INFORMASJON OM OPPSKYTNING")
    print(f"Det tok {t} sekunder ({t/60:.2f} minutter) å nå unnslipningshastighet.")
    print(f"Nåværende masse: {mass:.2f} kg")
    print(f"Drivstoff brukt: {(mass_initial - mass):.2f} kg")
    print(f"Gjenværende drivstoff: {(mass - 1100):.2f}")
    print()

    print(f"Rakettens fart relativt til oppskytningspunktet: ({v[0]}, {v[1]}) m/s")
    print(f"Rakettens posisjon relativt til oppskytningspunktet: ({r[0]}, {r[1]}) m")
    print()



    # --------------- Initialverdier til planeten ------------------
    print("Startposisjonene til Frogstar i astronomiske enheter:")
    print(f"({initial_positions[0][0]}, {initial_positions[1][0]})")
    print("Startfart til Frogstar i astronomiske enheter per år:")
    print(f"({initial_velocities[0][0]}, {initial_velocities[1][0]})")
    print()

    # ------------- Rakettens posisjon i forhold til sola ------------
    print(f"Avstand fra Frogstar til Star_Destroyer: {avstand_sol_planet_vec} km.")
    print(f"Rakettens posisjonskoordinater i forhold til Frogstar er {koordinater_rakett_oppskytningspunkt} km.")
    print(f"Rakettens posisjonskoordinater i forhold til sola er {koordinater_rakett_sol} km")
    print()


    # ----------------- Rakettens fart i forhold til sola ---------------   
    print(f"Farten til til planeten i forhold til sola er {v_planet_sol_vec} km/s.")
    print(f"Rakketens hastighetskoordinater i forhold til planeten er {v_rakett_jord} km/s.")
    print(f"Rakketens hastighetskoordinater i forhold til sola er {fart_rakett_sol} km/s")
    print()

    # ----------------- Space mission ---------------
    home_planet_idx = 0 # The home planet always has index 0
    print('My mission starts on planet {:d}, which has a radius of {:g} kilometers.'
      .format(home_planet_idx, mission.system.radii[home_planet_idx]))
    
    print('My spacecraft has a mass of {:g} kg and a cross-sectional area of {:g} m^2.'
      .format(mission.spacecraft_mass, mission.spacecraft_area))
    
    estimated_time = 987    # s
    mission.set_launch_parameters(F_motor, consumption, mass_initial_fuel, \
                                  estimated_time, avstand_sol_oppskytningspunkt_AU, 0)

    if not mission.rocket_launched:
        print('I have not launched the rocket yet. Let us do something about that!')
        mission.launch_rocket()

    mission.verify_launch_result(koordinater_rakett_sol_AU)



# Vi bommer på ca. 200 km ved oppskytningen. Vi antar at det er på grunn av 
# en feil i koordinatskifte



