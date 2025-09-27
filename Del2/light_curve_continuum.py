import ast2000tools.utils as utils
import ast2000tools.constants as const
seed = utils.get_seed('oafincke')
import numpy as np
import matplotlib.pyplot as plt

from ast2000tools.solar_system import SolarSystem
system = SolarSystem(seed)
from ast2000tools.space_mission import SpaceMission
mission = SpaceMission(seed)
import random

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
    T_tot = P_planet * 0.5   # total kjøretid i år
    time_steps_pr_year = 100000 
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

    return v_planet, v_star, dt

def light_curve():
    """Find the time-values and belonging flux-values for the relative flux."""
    # Definerer forskjellige variabler
    radius_planet = (system.radii[idx_planet])   # km
    radius_star = (system.star_radius)    # km
    area_star = np.pi * radius_star**2 # km^2

    v_planet, v_star, dt = LeapFrog()

    # Lager noen arrays og konverterer til ønskede enheter
    v_planet = utils.AU_pr_yr_to_m_pr_s(v_planet)*3.6  # km/t
    v_star = utils.AU_pr_yr_to_m_pr_s(v_star)*3.6   # km/t
    dt = utils.yr_to_s(dt) / 3600 # hours
    v_rel = v_planet - v_star   # km/t   

    # Definerer noen flere variabler
    travelled = 0
    flux_max = 1
    flux = np.zeros(485)    # tallet 485 fant vi ved å kjøre koden noen ganger
    time = np.zeros(485)
    i = 0
    diff = 0
    area_planet = 0
    buffer = 100000
    
    flux[0] = flux_max



    while travelled < buffer + 2*radius_star + 2*radius_planet + buffer:
        if travelled < buffer:
            # Fluksen er alltid bare 1
            flux[i+1] = flux_max
            time[i+1] = time[i] + dt

            travelled += np.linalg.norm(v_rel[i]) * dt
            i += 1
        elif travelled < buffer + 2 * radius_planet:
            # Beregner arealet som planeten har kommet foran sola med
            diff += np.linalg.norm(v_rel[i])*dt
            r_marked = radius_planet - diff
            h = radius_planet * np.sin(np.arccos(r_marked / radius_planet))
            area_planet += 2 * h * diff

            # Oppdaterer lister
            flux[i+1] = flux_max - area_planet/area_star
            time[i+1] = time[i] + dt
            np.linalg.norm(v_rel[i]) * dt
            travelled += np.linalg.norm(v_rel[i]) * dt
            i += 1
        elif travelled < buffer + 2 * radius_star:
            flux[i+1] = flux[i] #flux_min
            time[i+1] = time[i] + dt

            travelled += np.linalg.norm(v_rel[i]) * dt
            i += 1
            diff = 0
            area_planet = 0
        elif travelled < buffer + 2*radius_star + 2*radius_planet:
            # Regner ut arealet planeten har gått ut av sola med
            diff += np.linalg.norm(v_rel[i])*dt
            r_marked = radius_planet - diff
            h = radius_planet * np.sin(np.arccos(r_marked / radius_planet))
            area_planet += 2 * h * np.linalg.norm(v_rel[i])*dt

            # Oppdaterer lister
            flux[i+1] = flux_max - (np.pi*radius_planet**2 - area_planet)/area_star
            time[i+1] = time[i] + dt
            
            travelled += np.linalg.norm(v_rel[i]) * dt
            i += 1
        elif travelled < buffer + 2*radius_star + 2*radius_planet + buffer:
            # Nå er fluksen alltid bare 1
            flux[i+1] = flux_max
            time[i+1] = time[i] + dt

            travelled += np.linalg.norm(v_rel[i]) * dt
            i += 1

        # Legger på gaussisk støy på flux-verdier
        mean = 0
        sigma = 1e-4
        gauss_noise = np.random.normal(mean, sigma, (len(flux)))
        flux = flux + gauss_noise

    return flux, time



def plotter():
    """Plotter lyskurve med gaussisk støy"""
    fig, axs = plt.subplots(1, 1)
    flux, time = light_curve()
    axs.plot(time, flux, label = f"Light curve")
    plt.xticks(np.arange(0, time[-1], 2.5))
    axs.set_xlabel("Time [hours]")
    axs.set_ylabel("Relative flux")
    axs.set_title(f"Light curve of planet {idx_planet}")
    axs.grid()
    axs.legend(loc="upper right")
    plt.show()

if __name__ == "__main__":
    plotter()


# Her kunne vi åpenbart sluppet hishasset med å finne arealet som funksjon av
# tida ved å tilnærme fluksen fra punktet der planeten begynner å overlappe sola
# til den er helt innenfor med en rett linje. Da hadde vi sluppet alle if-tester.
# Til gjengjeld er det lettere å legge på gaussisk støy.