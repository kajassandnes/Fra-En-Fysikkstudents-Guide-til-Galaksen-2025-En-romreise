# Ikke brukt kodemal
# Program som plotter numeriske planetbaner over analytiske planetbaner
import numpy as np
from numba import njit

#print(numba.__version__)

import ast2000tools.utils as utils
import ast2000tools.constants as const
seed = utils.get_seed('oafincke')
import numpy as np
import matplotlib.pyplot as plt

from ast2000tools.solar_system import SolarSystem
system = SolarSystem(seed)
from ast2000tools.space_mission import SpaceMission
mission = SpaceMission(seed)


# ================ Øker størrelse på tekst på plots ====================
plt.rcParams['axes.labelsize'] = 26
plt.rcParams['axes.titlesize'] = 28
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 20


@njit
def akselerasjon(r_vector: np.ndarray[float], G:float, M:float) -> np.ndarray:
    """Finner akselerasjonen ved Newtons andre lov a = F/m der m er massen
    til sola og F er gravitasjonskraften mellom planeten og stjernen, 
    F = -(GM*r_hat)/r**2. Krafta er negativ fordi den er tiltrekkende og peker
    fra planeten mot stjernen, mens enhetsvektoren peker fra sola mot planeten.
    
    Parametere: 
    r_vec (np.ndarray[float]): posisjonen til planeten i x- og y-koordinater [AU]
    M (float): stjernemassen [i solmasser]
    G (float): gravitasjonskonstanten i AU-enheter
    r (float): absoluttverdien til r_vec [AU]
    
    Returnerer:
    aks (np.ndarray[float]): akselerasjon i x- og y-retning [AU/year²]
    """
    r = np.sqrt(r_vector[0]**2 + r_vector[1]**2)

    return - G*M / r**3 * r_vector

    
@njit
def numerisk_bane(r0, v0, G, M, P, runder=1):
    """Plotter de numeriske banene ved hjelp av leap-frog metoden.
    
    Parametere: 
    runder (float): default satt til 20 runder av hjemplaneten vår
    P (float): rundetid, default satt til perioden til hjemplaneten vår (s)
        
    T_tot (float): total kjøretid [years]
    time_step_pr_year (int): antall tidssteg per år
    time_steps (float): antall tidssteg
    dt (float): tidssteg (year)
    t (float): løpende tid [years]
        
    r_vec (ndarray(time_steps, 2)): beskriver posisjonen til planeten over 
                                    tid i x- og y-koordinater.
    v_vec: samme som r_vec bare med hastighet
    a_vec: samme som r_vec bare med akselerasjon
    i (int): teller

    Returnerer:
    r-vec (ndarray(time_steps, 2)): Planetens x- og y-koordinater over tid
    dt (float): tidssteg
    T_tot (float): total tid
    """
    T_tot = P * runder  
    time_steps_pr_year = 1e5
    time_steps = int(T_tot * time_steps_pr_year)
    dt = T_tot / time_steps

    # bevegelsesvektorer
    r_vec = np.zeros((time_steps + 1, 2))
    v_vec = np.zeros((time_steps + 1, 2))
    a_vec = np.zeros((time_steps + 1, 2))
        
    # initialverdier
    r_vec[0] = r0
    v_vec[0] = v0
    a_vec[0] = akselerasjon(r0, G, M)

    # Bruker Leap_Frog
    for i in range(time_steps):
        r_vec[i+1] = r_vec[i] + v_vec[i]*dt + 0.5*a_vec[i]*dt**2         
        a_vec[i+1] = akselerasjon(r_vec[i+1], G, M)
        v_vec[i+1] = v_vec[i] + 0.5*(a_vec[i] + a_vec[i+1])*dt

    return r_vec, v_vec, dt
    

def planet_bane(i:int, runder=1):
    """I er indeksen til planeten vi vil finne baneinformasjonen til"""
    G = const.G_sol
    M = system.star_mass 
    # rundetiden til planet 0. Det sørger for at dt er lik for alle planeter
    P = np.sqrt(4 * np.pi**2 * system.semi_major_axes[0]**3 / (const.G_sol * (system.star_mass + system.masses[0])))

    x, y = system.initial_positions[0][i], system.initial_positions[1][i]
    r0 = np.array([x, y])
    vx, vy = system.initial_velocities[0][i], system.initial_velocities[1][i]
    v0 = np.array([vx, vy])
                                                             
    r_vec, v_vec, dt = numerisk_bane(r0, v0, G, M, P, runder)    

    return r_vec, v_vec, dt 




if __name__ == "__main__":
    r, v, dt = planet_bane(0)
    plt.plot(r[:,0], r[:,1])
    plt.axis('equal')
    plt.show()