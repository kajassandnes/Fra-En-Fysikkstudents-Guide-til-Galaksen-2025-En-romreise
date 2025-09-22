# Ikke brukt kodemal
# Program som plotter numeriske planetbaner over analytiske planetbaner

import ast2000tools.utils as utils
import ast2000tools.constants as const
seed = utils.get_seed('oafincke')
import numpy as np
import matplotlib.pyplot as plt

from ast2000tools.solar_system import SolarSystem
system = SolarSystem(seed)
from ast2000tools.space_mission import SpaceMission
mission = SpaceMission(seed)

idx_planet = 4
def acceleration(r_rel: np.ndarray[float], mass: float) -> np.ndarray[float]:
    global idx_planet # indeksen til planeten vi velger å simulere
    """Beregner akselerasjonen til et legeme i et tolegemesystem ved bruk
    av Newtons andre lov: a = F/m.
    
    Parametere:
    r_rel (np.ndarray[float]): relativ avstand mellom de to legemene (AU)
    mass (float): massen til legemet som vi finner akselerasjonen til (solmasser)
    
    Return:
    a (np.ndarray[float]): akselerasjon i x- og y-retning (AU/år^2)
    """
    G = const.G_sol     # solenheter
    r = np.linalg.norm(r_rel)   # AU

    a = - G * mass / r**3 * r_rel
    return a



def LeapFrog() -> np.ndarray[float]:
    """Simulerer stjernen og en planet i solsystemet som et tolegemesystem med
    referansepunkt i origo og et driftende massesenter.
    """
    global idx_planet
    # ---------------- Konstanter -----------------
    mass_star = system.star_mass   # solar mass
    mass_planet = system.masses[idx_planet] # solar mass
    G = const.G_sol     # Gravitational constant in AU
    a = system.semi_major_axes[idx_planet]  # AU
    # --------------- tidsinstillinger -------------------
    P_planet = np.sqrt(4*np.pi**2 * a**3 / (G * (mass_star + mass_planet)))    # perioden til planeten i år
    T_tot = P_planet * 10   # total kjøretid i år
    time_steps_pr_year = 10000 
    time_steps = int(T_tot * time_steps_pr_year)
    dt = T_tot / time_steps
    t = dt
    # ----------------------------------------------------

    # ----------- initialbetingelser -------------
    r_star = np.zeros((time_steps + 1, 2))
    v_star = np.zeros((time_steps + 1, 2))
    a_star = np.zeros((time_steps + 1, 2))

    r_planet = np.zeros((time_steps + 1, 2))
    v_planet = np.zeros((time_steps + 1, 2))
    a_planet = np.zeros((time_steps + 1, 2)) 

    r_planet[0] = np.array([system.initial_positions[0][idx_planet], system.initial_positions[1][idx_planet]])
    v_planet[0] = np.array([system.initial_velocities[0][idx_planet], system.initial_velocities[1][idx_planet]])

    r_rel = r_star[0] - r_planet[0]  # fra planeten til stjerna

    a_planet[0] = acceleration(r_rel, mass_planet)

    r_star[0] = np.array([0, 0])
    v_star[0] = np.array([0, 0])
    a_star[0] = acceleration(r_rel, mass_star)
    # -----------------------------------------
    
    i = 0

    # Bruker Leap_Frog
    while t < T_tot:  
        r_planet[i+1] = r_planet[i] + v_planet[i]*dt + 0.5*a_planet[i]*dt**2
        r_star[i+1] = r_star[i] + v_star[i]*dt + 0.5*a_star[i]*dt**2   

        r_rel = r_star[i] - r_planet[i]

        a_planet[i+1] = - acceleration(r_rel, mass_star) # Negative because acceleration is opposit to unit vector
        a_star[i+1] = acceleration(r_rel, mass_planet)   

        v_planet[i+1] = v_planet[i] + 0.5*(a_planet[i] + a_planet[i+1])*dt
        v_star[i+1] = v_star[i] + 0.5*(a_star[i] + a_star[i+1])*dt
    
        t += dt
        i += 1


    return r_planet, r_star


def plotter():
    fig, axs = plt.subplots(1, 1)
    r_planet, r_star = LeapFrog()
    axs.plot(r_planet[:,0], r_planet[:,1], label = f"Planet nr. {idx_planet}")
    axs.plot(r_star[:,0], r_star[:,1], label = f"Star")
    axs.set_xlabel("Position x-axis [AU]")
    axs.set_ylabel("Position y-axis [AU]")
    axs.set_title("Two body problem")
    axs.axis("equal")
    axs.grid()
    axs.legend(loc="upper right")
    plt.show()

if __name__ == "__main__":
    plotter()


