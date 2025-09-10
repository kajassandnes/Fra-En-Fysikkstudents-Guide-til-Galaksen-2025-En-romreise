# Ikke brukt kodemal
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

# ============ Rotasjonsfart =============
rotasjon_dager = system.rotational_periods      # dager / rotasjon
rotasjon_sekunder = utils.day_to_s(rotasjon_dager[0]) # sekunder / rotasjon
vinkelfart = 2*np.pi/rotasjon_sekunder  # radianer / sekund
rotasjonsfart_m_pr_sek = (vinkelfart * R_0)   # m / sekund
# ========================================

# ======== Unnslipningsfart ===============
v_escape = np.sqrt(2*G*M_0/R_0)     # m/s
v_initial = rotasjonsfart_m_pr_sek   # m/s  
v_rest = np.sqrt(v_escape**2 - v_initial**2)  # m/s
# ============================================

# ======== Bevegelsesrelevante arrays ==========
r = np.array([0., 0.])
v = np.array([0., v_initial])
a = np.array([0., 0.])
# =============================================

# =========== Relevant til løkka =============
t = 0   # s
dt = 0.1      # sekund 
F_motor = 60000       # N
mass_initial = 1100 + 6500  # rakettmasse + drivstoffmasse
mass = 7600 # kg
consumption = 6.0   # kg/s

# Følgende lister lagrer verdier til plot
v_historie = []
a_historie = []
r_historie = []
# ============================================


def tyngdeakselerasjon(r):
    """Regner ut tyngdeakselerasjonen Frogstar drar raketten vår nedover
    med basert på rakettens nåværende distanse fra massesenteret.
    """
    global R_0, G, M_0
    R = R_0 + np.linalg.norm(r)
    a = G*M_0 / R**2
    return a

def launch():
    """Bruker Eulers metode til å regne ut ny fart og posisjon til raketten.
    akselerasjonen regnes ut ved å ta akselerasjonen til raketten og 
    trekke fra tyngdeakselerasjonen. Oppskytningen avsluttes når raketten har 
    oppnår unnslipningshastighet.
  
    Funksjonen lagrer også historien til absoluttverdiene til akselerasjonen 
    og farten, slik at de kan plottes mot tid. Posisjonen lagres slik at 
    man kan plotte x-verdier og y-verdier mot hverandre."""
    global r, v, a, v_escape, t, dt, mass, consumption, F_motor

    while np.linalg.norm(v) < v_escape:
        a[0] = F_motor/mass - tyngdeakselerasjon(r)
        v += a * dt
        r += v * dt

        t += dt

        a_historie.append(np.linalg.norm(a))
        v_historie.append(np.linalg.norm(v)) 
        r_historie.append(np.array([r[0], r[1]]))

        if mass < 1100:
            return "Du er desverre tom for drivstoff"
        
        mass -= consumption * dt


if __name__ == "__main__":
    launch()
    # ========== Beregninger ===========
    tid_til_unnslipningsfart = np.arange(0, t-dt, dt)
    r_historie = np.array(r_historie)

    initial_positions = system.initial_positions
    initial_velocities = system.initial_velocities

    # ---------- Rakketens posisjon i forhold til sola ----------------
    planetradien = np.array([system.radii[0], 0])      # km
    avstand_sol_planet_x = utils.AU_to_km(initial_positions[0][0])  # km
    avstand_sol_planet_y = utils.AU_to_km(initial_positions[1][0])  # km
    avstand_sol_planet_initial = np.array([avstand_sol_planet_x, avstand_sol_planet_y])  # km
    koordinater_rakett_planetsystem = r * 1e-3  # km

    koordinater_rakett_solsystem = avstand_sol_planet_initial + planetradien + koordinater_rakett_planetsystem  # km

    # ------------ Rakketens fart i forhold til sola
    fart_rakett_jordsystem = v * 1e-3 # km/s 
    fart_planet_sol_x = utils.AU_pr_yr_to_m_pr_s(initial_velocities[0][0]) / 1000  # km/s
    fart_planet_sol_y = utils.AU_pr_yr_to_m_pr_s(initial_velocities[1][0]) / 1000   # km/s
    fart_planet_sol_vec = np.array([fart_planet_sol_x, fart_planet_sol_y])      # km / s

    fart_rakett_sol = fart_planet_sol_vec + fart_rakett_jordsystem


    # --------- printer informasjon -----------
    print(f"Rotasjonsfarten til Frogstar: {rotasjonsfart_m_pr_sek} m/s i y-retning")
    print()

    print(f"Unnslipningsfart: {v_escape} m/s")
    print(f"Initialfart: {v_initial} m/s")
    print(f"Vi må akselerere til en fart på: {v_rest} m/s")
    print()

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


    # --------------- Informasjon om oppskytninga -----------------
    print(f"Det tok {t} sekunder ({t/60} minutter) å nå unnslipningshastighet.")
    print(f"Nåværende masse: {mass} kg")
    print(f"Drivstoff brukt: {mass_initial - mass} kg")
    print(f"Gjenværende drivstoff: {mass - 1100}")
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
    print(f"Avstand fra Frogstar til Star_Destroyer: {avstand_sol_planet_initial} km.")
    print(f"Rakettens posisjonskoordinater i forhold til Frogstar er {koordinater_rakett_planetsystem} km.")
    print(f"Rakettens posisjonskoordinater i forhold til sola er {koordinater_rakett_solsystem} km")
    print()


    # ----------------- Rakettens fart i forhold til sola ---------------   
    print(f"Farten til til planeten i forhold til sola er {fart_planet_sol_vec} km/s.")
    print(f"Rakketens hastighetskoordinater i forhold til planeten er {fart_rakett_jordsystem} km/s.")
    print(f"Rakketens hastighetskoordinater i forhold til sola er {fart_rakett_sol} km/s")
    print()



