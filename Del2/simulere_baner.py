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

# ============================== Planet klasse ================================
class Planet():
    def __init__(self, semi_major_axes, eccentricity, planet_mass, planet_radius, \
                 aphelion_angle, init_orbit_angle, x, y, vx ,vy, nr):
        self._a = semi_major_axes    # AU
        self._e = eccentricity   
        self._mass = planet_mass     # mass comparred to sun
        self._radius = planet_radius     # km
        self._m = const.G_sol * (planet_mass + system.star_mass)  # G(m_planet + m_sun)
        self._ang = aphelion_angle   # angle from x-axes to point farthest from the sun (radians)
        self._init_ang = init_orbit_angle    # (radians)
        self._P = np.sqrt(4*np.pi**2*self._a**3 / self._m)

        self.x = x  # AU
        self.y = y  # AU
        self.vx = vx    # AU / year
        self.vy = vy    # AU / year 
        self.r = np.array([self.x, self.y])
        self.v = np.array([self.vx, self.vy])
        self.a = np.zeros(2)
        self.current_angle = 0

        self._nr = nr
        self._h = np.linalg.norm([0, 0, x*vy - y*vx])
        self._p = self._h**2 / self._m


    
    def akselerasjon(self, r_vector: np.ndarray[float]):
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
        M = system.star_mass       # i solenheter
        r = np.linalg.norm(r_vector)

        aks = - G*M / r**3 * r_vector
        return aks

    
    def numerisk_bane(self, runder=20):
        """Plotter de numeriske banene ved hjelp av leap-frog metoden.
        
        Parametere: 
        t (float): løpende tid [years]
        T_tot (float): total kjøretid [years]
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
        P = np.sqrt(4 * np.pi**2 * system.semi_major_axes[0]**3 / (const.G_sol * (system.star_mass + system.masses[0])))
        T_tot = P * runder  
        time_steps_pr_year = 10000 
        time_steps = int(T_tot * time_steps_pr_year)
        dt = T_tot / time_steps
        t = dt

        r_vec = np.zeros((time_steps + 1, 2))
        v_vec = np.zeros((time_steps + 1, 2))
        a_vec = np.zeros((time_steps + 1, 2))

        r_vec[0] = self.r
        v_vec[0] = self.v
        a_vec[0] = np.array([self.akselerasjon(self.r)])

        i = 0

        # Bruker Leap_Frog
        while t < T_tot:  
            r_vec[i+1] = r_vec[i] + v_vec[i]*dt + 0.5*a_vec[i]*dt**2         
            a_vec[i+1] = self.akselerasjon(r_vec[i+1])
            v_vec[i+1] = v_vec[i] + 0.5*(a_vec[i] + a_vec[i+1])*dt
     
            t += dt
            i += 1


        return r_vec, dt
    

    def analytical_orbits(self):
        d_theta = 0.1
        theta_values = np.arange(self._init_ang, self._init_ang + 2 * np.pi + d_theta, d_theta)
        r_vec = np.zeros((len(theta_values),2))

        for i, theta in enumerate(theta_values):
            f = theta - (self._ang - np.pi)
            r = self._p / (1 + (self._e * np.cos(f))) #likning fra forelesningsnotater
            r_vec[i, 0] = r*np.cos(theta)
            r_vec[i, 1] = r*np.sin(theta)

        return r_vec
    
    
    def analytical_orbit_plotter(self):
        r = self.analytical_orbits()
        plt.plot(r[:,0], r[:,1], label = f"Planet nr. {self._nr} analytisk")
        plt.xlabel("posisjon langs x [AU]")
        plt.ylabel("posisjon langs y [AU]")
        plt.title("Analytiske baner")
        plt.grid()
        plt.legend()


    def numerical_orbit_plotter(self):
        r, dt = self.numerisk_bane()
        plt.plot(r[:,0], r[:,1], label = f"Planet nr. {self._nr} numerisk")
        plt.xlabel("posisjon langs x [AU]")
        plt.ylabel("posisjon langs y [AU]")
        plt.title("Numeriske baner")
        plt.grid()
        plt.legend()


    def lite_areal(self, u, v):
        """Arealet til en trekant er gitt ved (1/2)*|r1 x r2|"""
        areal = 0.5 * np.abs(np.cross(u, v))
        return areal
    
    def stort_areal(self):
        t_tot = self._P / 10
        r_vec, dt = self.numerisk_bane(1)   

        idx_max = np.argmax(np.linalg.norm(r_vec, axis=1))
        idx_min = np.argmin(np.linalg.norm(r_vec, axis=1))

        time_steps = int(t_tot // dt)

        
        r_max_vec = r_vec[idx_max : idx_max + (time_steps + 1)]
        r_min_vec = r_vec[idx_min : idx_min + (time_steps + 1)]


        areal_max = 0
        areal_min = 0

        R1 = 0
        R2 = 0

        # Vektoren fra sola til planeten ved t = 0 er r, fra sola til planeten
        # ved t = 1 er r'. Vektoren fra r til r' er dr = r' - r
        for i in range(len(r_max_vec) - 1): 
            u1 = r_max_vec[i]
            v1 = r_max_vec[i+1]
            R1 += np.linalg.norm(v1 - u1)
            areal_max += self.lite_areal(u1, v1)

            u2 = r_min_vec[i]
            v2 = r_min_vec[i+1]
            R2 += np.linalg.norm(v2 - u2)
            areal_min += self.lite_areal(u2, v2)

        mean_velocity_1 = R1 / t_tot
        mean_velocity_2 = R2 / t_tot

        difference = abs(areal_max - areal_min)
        relative_uncertainty = difference / areal_max

        return areal_max, areal_min, difference, relative_uncertainty, R1, R2, mean_velocity_1, mean_velocity_2
    




def Kepler(planet):
    return planet.stort_areal()

def plot_analytical_orbits(planet_objects):
    for planet in planet_objects:
        planet.analytical_orbit_plotter() 
    
    plt.show()

def plot_numerical_orbits(planet_objects):
    for planet in planet_objects:
        planet.numerical_orbit_plotter() 
    
    plt.show()

def plot_numerical_and_analytical_orbits(planet_objects):
    for planet in planet_objects:
        planet.numerical_orbit_plotter() 
        planet.analytical_orbit_plotter()
    
    plt.show()


def create_all_planet_objects():
    planet_objekter = []
    for i in range(8):
        planet_objekt = Planet(system.semi_major_axes[i], system.eccentricities[i], \
                system.masses[i], system.radii[i], system.aphelion_angles[i], \
                system.initial_orbital_angles[i], \
                system.initial_positions[0][i], system.initial_positions[1][i], \
                system.initial_velocities[0][i], system.initial_velocities[1][i], i)
        
        planet_objekter.append(planet_objekt)
    
    return planet_objekter



if __name__ == "__main__":
    all_planet_objects = create_all_planet_objects()
    #plot_numerical_orbits(all_planet_objects)
    #plot_analytical_orbits(all_planet_objects)
    #plot_numerical_and_analytical_orbits(all_planet_objects)

    aphelion_area, perihelion_area, abs_uncertainty, rel_uncertainty, distance_aph, distance_perih, mean_velocity_aph, mean_velocity_perih = Kepler(all_planet_objects[0])
    print()
    print("ANALYSE AV KEPLERS 2. LOV")
    print(f"Areal sveipet ut på t=P/10 år nærmest ved perihelion: {perihelion_area} AU")
    print(f"Areal sveipet ut på t =P/10 lengst unna ved aphelion: {aphelion_area} AU")
    print(f"Differanse, absolutt usikkerhet: {abs_uncertainty} AU")
    print(f"Relativ usikkerhet: {rel_uncertainty}")
    print(f"Prosentvis andel perihelion av aphelion: {perihelion_area/aphelion_area * 100} %")
    print()
    print(f"Distance travelled aphelion: {distance_aph} AU")
    print(f"Distance travelled perihelion: {distance_perih} AU")
    print(f"Mean velocity at aphelion: {mean_velocity_aph} AU/year")
    print(f"Mean velocity at perihelion: {mean_velocity_perih} AU/year")





# Arealet som er sveipet ut er rimelige størrelser i forhold til jorda
    

