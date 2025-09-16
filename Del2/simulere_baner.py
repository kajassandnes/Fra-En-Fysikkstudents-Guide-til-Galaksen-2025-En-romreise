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

# ============= NUMERISKE PLOT ================

class Planet():
    def __init__(self, semi_major_axes, eccentricity, planet_mass, planet_radius, aphelion_angle, init_orbit_angle, x , y, vx ,vy, nr):
        self._a = semi_major_axes    # AU
        self._e = eccentricity   
        self._mass = planet_mass     # mass comparred to sun
        self._radius = planet_radius     # km
        self._m = const.G_sol * (planet_mass + system.star_mass)  # G(m_planet + m_sun)
        self._ang = aphelion_angle   # angle from x-axes to point farthest from the sun (radians)
        self._init_ang = init_orbit_angle    # (radians)
        self.x = x  # AU
        self.y = y  # AU
        self.vx = vx    # AU / year
        self.vy = vy    # AU / year 
        self.r = np.array([self.x,self.y])
        self.v = np.array([self.vx, self.vy])
        self._nr = nr
        self._h =  np.linalg.norm(np.array([x,y])) * np.linalg.norm(np.array([vx,vy])) * np.cos(self._init_ang) #angular momentum
        self._p = self._h**2 / self._m


    
    def akselerasjon(self, r_vec: np.ndarray[float]):
        """Finner akselerasjonen ved Newtons andre lov a = F/m der m er massen
        til sola og F er gravitasjonskraften mellom planeten og stjernen, 
        F = -GM*r_hat/r**2. Krafta er negativ fordi den er tiltrekkende og peker
        fra planeten mot stjernen, mensc enhetsvektoren peker fra sola mot planeten.
        
        Parametere: 
        r_vec (np.ndarray[float]): posisjonen til planeten i x- og y-koordinater [AU]
        M (float): stjernemassen [i solmasser]
        G (float): gravitasjonskonstanten i AU-enheter
        r (float): absoluttverdien til r_vec [AU]
        
        Returnerer:
        aks (np.ndarray[float]): akselerasjon i x- og y-retning [AU/year²]
        """
        G = const.G_sol
        M = const. m_sun        # i solenheter
        r = np.linalg.norm(r_vec)
        aks = -G*M / r * r_vec
        return aks

    
    def numerisk_bane(self):
        """Plotter de numeriske banene ved hjelp av leap-frog metoden.
        
        Parametere: 
        t (float): løpende tid [years]
        T_tot (float): total tid [years]
        time_step_pr_year (int): antall tidssteg per år
        dt (float): tidssteg
        N (float): antall tidssteg
        self.r (np.ndarray[np.ndarray[float]]): array som beskriver posisjonen
        til planeten der hvert element er en array med x- og y-koordinater.
        self.v: samme som self.r bare med hastighet
        self.a: samme som self.r bare med akselerasjon
        i (int): teller

        Returnerer:
        self.r: Array med x- og y-koordinater som skal plottes over analytiske 
        baner.
        """
        t = 0   # years
        T_tot = 1  # years
        time_step_pr_year = 10000
        dt = T_tot / time_step_pr_year
        N = time_step_pr_year*T_tot + 2

        self.r = np.zeros((N, 2))
        self.v = np.zeros((N, 2))
        self.a = np.zeros((N, 2))
        
        self.r[0] = np.array([self.x, self.y])
        self.v[0] = np.array([self.vx, self.vy])
        self.a[0] = np.array([self.akselerasjon(self.r[0])])

        i = 0

        while t < T_tot:
            self.a[i+1] = self.akselerasjon(self.r[i])
            self.v[i+1] = self.v[i] + 0.5*(self.a[i] + self.a[i+1])*dt
            self.r[i+1] = self.r[i] + self.v[i]*dt + 0.5*self.a[i]*dt**2
            
            t += dt
            i += 1

        return self.r
    
    def plotter(self):
        r = self.numerisk_bane()
        plt.plot(self.r[:,0], self.r[:,1], label = f"Planet nr. {self._nr}")
        plt.xlabel("posisjon langs x [AU]")
        plt.ylabel("posisjon langs y [AU]")
        plt.title("Numeriske baner")
        plt.grid()
        plt.legend()
        plt.show()
        
    

def create_planet_objects():
    planet_objekter = []
    for i in range(8):
        planet_objekt = Planet(system.semi_major_axes[i], system.eccentricities[i], \
                system.masses[i], system.radii[i], system.aphelion_angles[i], \
                system.initial_orbital_angles[i], \
                system.initial_positions[0][i], system.initial_positions[1][i], \
                system.initial_velocities[0][i], system.initial_velocities[1][i], i)
        
        planet_objekter.append(planet_objekt)
    
    return planet_objekter


planet_objects = create_planet_objects()

#planet_objects[0].plotter()

print("HEI")