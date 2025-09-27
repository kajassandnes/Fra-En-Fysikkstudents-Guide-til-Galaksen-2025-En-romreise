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


# ================ Øker størrelse på tekst på plots ====================
plt.rcParams['axes.labelsize'] = 26
plt.rcParams['axes.titlesize'] = 28
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 20


# ============================== Planet klasse ================================
class Planet():
    """Klasse som representerer planeter og ulike beregninger relatert til dem."""

    def __init__(self, semi_major_axis: float, eccentricity: float, planet_mass:float, \
                 planet_radius: float, aphelion_angle: float, init_orbit_angle:float, \
                 x: float, y: float, vx: float ,vy: float, nr: int):
        """Konstruktør
        Atributter:
        _a (float): semi-major-axis of ellipse (AU)
        _e (float): eccentricity of ellipse (0-1)
        _mass (float): planet mass (solar-masses)
        _radius (float): planet radius (km)
        _m (float): G*(M+m) (gravitational constant*(star mass + planet mass)) (AU^3/yr^2)
        _ang (float): aphelion angle, from x-axis to point farthest from the sun (radians)
        _init_ang (float): initial angle of planet (radians)
        _P (float): period of planet - using Newtons corrected version og Keplers 3rd law (s)

        r (ndarray): position of planet in x- and y-coordinate (AU)
        v (ndarray): velocity of planet in x- and y-coordinate (AU/year)
        a (ndarray): acceleration of planet in x- and y-coordinate (AU/year^2)

        _nr (int): number of planet in the solar system (0-7)
        _h (float): angular momentum per mass - constant value (AU^2/year)
        _p (float): constant value in formula for ellipse (AU)
        """
        self._a = semi_major_axis    
        self._e = eccentricity   
        self._mass = planet_mass   
        self._radius = planet_radius     
        self._m = const.G_sol * (planet_mass + system.star_mass)  
        self._ang = aphelion_angle   
        self._init_ang = init_orbit_angle   
        self._P = np.sqrt(4*np.pi**2*self._a**3 / self._m)

        self.r = np.array([x, y])
        self.v = np.array([vx, vy])
        self.a = np.zeros(2)

        self._nr = nr
        self._h = np.linalg.norm([0, 0, x*vy - y*vx])
        self._p = self._h**2 / self._m


    
    def akselerasjon(self, r_vector: np.ndarray[float]) -> np.ndarray:
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
        G = const.G_sol
        M = system.star_mass       # i solenheter
        r = np.linalg.norm(r_vector)

        aks = - G*M / r**3 * r_vector
        return aks

    
    def numerisk_bane(self, runder=20, P=np.sqrt(4 * np.pi**2 * system.semi_major_axes[0]**3 \
                    / (const.G_sol * (system.star_mass + system.masses[0]))) \
                    ) -> np.ndarray:
        """Plotter de numeriske banene ved hjelp av leap-frog metoden.
        
        Parametere: 
        runder (float): default satt til 20
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
        time_steps_pr_year = 10000 
        time_steps = int(T_tot * time_steps_pr_year)
        dt = T_tot / time_steps
        t = dt

        # bevegelsesvektorer
        r_vec = np.zeros((time_steps + 1, 2))
        v_vec = np.zeros((time_steps + 1, 2))
        a_vec = np.zeros((time_steps + 1, 2))
        
        # initialverdier
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

        return r_vec, dt, T_tot
    

    def analytical_orbits(self):
        """Metode som beregner ellipsebanen til planetene analytisk én runde.
        Metoden går gjennom alle vinkler i enhetssirkelen og regner ut posisjon
        som funksjon av vinkel ved det analytiske uttrykket for ellipsebaner.
        
        Parametere:
        d_theta (float): endring i vinkel (radianer)
        theta_values (ndarray): vinkler fra startvinkel til +2pi radianer (radianer)
        r_vec (ndarray): posisjon som funksjon av vinkel i x- og y-koordinater

        f (float): theta - omega, der omega er perihelion vinkel (derfor 'aphelion - pi')
        r (float): analytisk løsning av ellipsebane
                   x = r*cos(theta)
                   y = r*sin(theta)  
        
        Returnerer:
        r_vec (ndarray): alle posisjoner som funksjon av vinkel
        """
        d_theta = 0.1
        theta_values = np.arange(self._init_ang, self._init_ang + 2 * np.pi + d_theta, d_theta)
        r_vec = np.zeros((len(theta_values), 2))

        for i, theta in enumerate(theta_values):
            f = theta - (self._ang - np.pi)
            r = self._p / (1 + (self._e * np.cos(f))) #likning fra forelesningsnotater
            r_vec[i, 0] = r*np.cos(theta)
            r_vec[i, 1] = r*np.sin(theta)

        return r_vec
    
    
    def analytical_orbit_plotter(self, axs):
        """Lager plot med analytiske baner"""
        r = self.analytical_orbits()
        axs.plot(r[:,0], r[:,1], label = f"Planet nr. {self._nr}")
        axs.set_xlabel("Position x-axis [AU]")
        axs.set_ylabel("posisjon y-axis [AU]")
        axs.set_title("Analytical orbits")
        axs.axis("equal")
        axs.grid()
        axs.legend(loc="upper right")


    def numerical_orbit_plotter(self, axs):
        """Lager plot med numeriske baner"""
        r, dt, P = self.numerisk_bane()
        axs.plot(r[:,0], r[:,1], label = f"Planet nr. {self._nr}")
        axs.set_xlabel("Position x-axis [AU]")
        axs.set_ylabel("Position y-axis [AU]")
        axs.set_title("Numerical orbits")
        axs.axis("equal")
        axs.grid()
        axs.legend(loc="upper right")

    def numerical_and_analytical_plotter(self, axs):
        """Lager ett plot med numeriske og analytiske baner"""
        r_num, dt, P = self.numerisk_bane()
        r_anal = self.analytical_orbits()
        axs.plot(r_num[:,0], r_num[:,1])
        axs.plot(r_anal[:,0], r_anal[:,1])
        axs.set_xlabel("Position x-axis [AU]")
        axs.set_ylabel("Position y-axis [AU]")
        axs.set_title("Numerical and analytical orbits")
        axs.axis("equal")
        axs.grid()
        axs.legend(loc="upper right")
        


    def lite_areal(self, u, v):
        """Arealet til en trekant er gitt ved (1/2)*|r1 x r2|"""
        areal = 0.5 * np.abs(np.cross(u, v))
        return areal
    
    def stort_areal(self):
        """Beregner arealet en planet sveiper ut for en tid lik 1/10 av dens 
        periode, nær aphelion og nær perihelion.
        
        Parametere:
        t_tot (float): totaltiden er en tiendedel av perioden
        idx_max (int): indeksen til elementet med størst avstand (finner aphelion)
        idx_min (int): indeksen til elementet med minst avstand (finner perihelion)

        Lager to nye arrays som er et utsnitt av simulasjonsarrayen, som inneholder
        alle posisjonene til planet 0 for én runde rundt stjernen. 
        Den ene er nær perihelion, den andre er nær aphelion

        Regner ut arealet sveipet ut per tidsenhet for en tid t_tot, strekningen
        som ble tilbakelagt på denne tida og gjennomsnittsfarten.

        Returnerer:
        areal_max: areal ved aphelion
        areal_min: areal ved perihelion
        difference: absolutt usikkerhet
        relative_uncertainty: relativ usikkerhet
        R1: total strekning reist ved aphelion
        R2: total strekning reist ved perihelion
        mean_velocity_max: gjennomsnittsfart ved aphelion
        mean_velocity_min: gjennomsnittsfart ved perihelion
        """
        t_tot = self._P / 10
        r_vec, dt, T_tot = self.numerisk_bane(1)   
        time_steps = int(t_tot // dt)

        # indekser til aphelion og perihelion
        idx_max = np.argmax(np.linalg.norm(r_vec, axis=1))
        idx_min = np.argmin(np.linalg.norm(r_vec, axis=1))

        # utsnitt av posisjoner nær aphelion og perihelion
        r_max_vec = r_vec[idx_max : idx_max + (time_steps + 1)]
        r_min_vec = r_vec[idx_min : idx_min + (time_steps + 1)]

        # startverdier
        areal_max = 0
        areal_min = 0
        R1 = 0
        R2 = 0

        # Vektoren fra sola til planeten ved t = 0 er u1, fra sola til planeten
        # ved t = t + dt er v1. Vektoren fra u1 til v1 er dr = v1 - u1
        # arealet er en halv ganger kryssproduktet av de to (en trekant)
        for i in range(len(r_max_vec) - 1): 
            u1 = r_max_vec[i]
            v1 = r_max_vec[i+1]
            R1 += np.linalg.norm(v1 - u1)
            areal_max += self.lite_areal(u1, v1)

            u2 = r_min_vec[i]
            v2 = r_min_vec[i+1]
            R2 += np.linalg.norm(v2 - u2)
            areal_min += self.lite_areal(u2, v2)

        # v = s/t
        mean_velocity_max = R1 / t_tot
        mean_velocity_min = R2 / t_tot

        # usikkerheter
        difference = abs(areal_max - areal_min)
        relative_uncertainty = difference / areal_max

        return areal_max, areal_min, difference, relative_uncertainty, R1, R2, mean_velocity_max, mean_velocity_min
    
    
    def numerisk_periode(self):
        """Regner ut perioden til alle planetene ved å sjekke tida det tar for 
        hver enkelt planet å gå én runde rundt stjerna.
        Metoden bruker Leapfrog til å regne banen numerisk og stopper når planeten
        har gått én runde. Da er tiden også telt opp.
        
        Parametere:
        theta (float): løpende vinkel som går fra initial-vinkel til 2pi mer
        t (float): løpende tid (s)
        dt (float): tidssteg (s)

        r (ndarray): éndimensjonal array for posisjon med x- og y-koordinat
        v (ndarray): samme som r for fart
        a (ndarray): samme som r for akselerasjon

        Returnerer: 
        P (float): periodetid = t
        """
        theta = self._init_ang
        t = 0
        dt = 0.0001

        r = self.r
        v = self.v
        a = self.akselerasjon(r)

        # Bruker Leap_Frog
        while theta < self._init_ang + 2*np.pi:
            r_ny = r + v*dt + 0.5*a*dt**2
            a_ny = self.akselerasjon(r)
            v = v + 0.5*(a + a_ny)*dt

            # regner vinkelen som blir spent ut av linja fra sola til r(t)
            # og r(t+dt)
            theta += np.linalg.norm(r_ny - r) / np.linalg.norm(r)

            a = a_ny
            r = r_ny
            
            t += dt
        
        P = t

        return P




def Kepler(planet):
    """Returnerer arealet sveipet ut av en ønsket planet ved perihelion og 
    aphelion"""
    return planet.stort_areal()


def plot_analytical_orbits(planet_objects):
    """Sørger for at de analytiske banene til alle planetene blir plottet
    samtidig."""
    fig, axs = plt.subplots(1, 1)
    for planet in planet_objects:
        planet.analytical_orbit_plotter(axs) 
    
    plt.show()

def plot_numerical_orbits(planet_objects):
    """Sørger for at de numeriske banene til alle planetene blir plottet
    samtidig."""
    fig, axs = plt.subplots(1, 1)
    for planet in planet_objects:
        planet.numerical_orbit_plotter(axs) 
    
    plt.show()

def plot_numerical_and_analytical_orbits(planet_objects):
    """Sørger for at de analytiske og numeriske banene til alle planetene 
    blir plottet samtidig."""
    fig, axs = plt.subplots(1, 1)
    for planet in planet_objects:
        planet.numerical_and_analytical_plotter(axs)
    
    plt.show()


def periodetider(planet_objects):
    """Samler data om periodetidene til alle planetene og sammenligner dem.
    
    I for-løkka regnes periodetiden for hver planet ut ved å bruke 
    Keplers 3. lov, Newtons versjon og numerisk utregning.
    
    Returnerer arrays med periodetidene til alle planetene utregnet på
    de forskjellige måtene, samt forholdet mellom Keplers og Newtons versjon.
    """
    periodetidene = np.zeros(8)
    period_kepler = np.zeros(8)
    period_newton = np.zeros(8)
    proposionality = np.zeros(8)

    i = 0
    for object in planet_objects:
        P = object.numerisk_periode()
        p_kepler = np.sqrt(object._a**3) # keplers 3. lov
        p_newton = np.sqrt((4*np.pi**2*object._a**3)/object._m) # Newtons versjon

        # lagrer verdiene i riktig array
        periodetidene[i] = P
        period_kepler[i] = p_kepler
        period_newton[i] = p_newton
        proposionality[i] = period_newton[i] / period_kepler[i]
        
        i += 1
    
    return periodetidene, period_kepler, period_newton, proposionality



def create_all_planet_objects():
    """Lager en liste som inneholder alle planetobjekter."""
    planet_objekter = []

    for i in range(8):
        planet_objekt = Planet(system.semi_major_axes[i], system.eccentricities[i], \
                system.masses[i], system.radii[i], system.aphelion_angles[i], \
                system.initial_orbital_angles[i], \
                system.initial_positions[0][i], system.initial_positions[1][i], \
                system.initial_velocities[0][i], system.initial_velocities[1][i], i)
        
        planet_objekter.append(planet_objekt)
    
    return planet_objekter


def get_information():
    all_planet_objects = create_all_planet_objects()
    
    number_of_times = 154862
    times = np.zeros(number_of_times)
    planet_positions = np.zeros((2, 8, number_of_times))
    for i in range(len(all_planet_objects)):
        r_planet, dt, T_tot = all_planet_objects[i].numerisk_bane()
        planet_positions[0][i] = r_planet[:,0]
        planet_positions[1][i] = r_planet[:,1]
    
    for i in range(len(planet_positions[0][0])-1):
        times[i+1] = times[i] + dt

    print(len(planet_positions[0][0])-1)
    return planet_positions, T_tot+0.0001, times


def plot_information() -> str:
    all_planet_objects = create_all_planet_objects()
    plot_numerical_orbits(all_planet_objects)
    plot_analytical_orbits(all_planet_objects)
    plot_numerical_and_analytical_orbits(all_planet_objects)

    aphelion_area, perihelion_area, abs_uncertainty, rel_uncertainty, distance_aph, distance_perih, mean_velocity_aph, mean_velocity_perih = Kepler(all_planet_objects[0])
    areal_velocity = aphelion_area / (all_planet_objects[0]._P/10)
    print()
    print("ANALYSE AV KEPLERS 2. LOV")
    print(f"Areal velocity: {areal_velocity}")
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

    periodetidene, period_kepler, period_newton, proporsonality = periodetider(all_planet_objects)
    for i in range(len(periodetidene)):
        print(f"Periodetiden til planet {i}: {periodetidene[i]} years")
    print()
    for i in range(len(periodetidene)):
        print(f"P^2 Kepler til planet {i}: {period_kepler[i]} years")
    print()
    for i in range(len(periodetidene)):
        print(f"P^2 Newton til planet {i}: {period_newton[i]} years")
    print()
    for i in range(len(periodetidene)):
        print(f"Proposjonalitet til planet {i}: {proporsonality[i]}")

    relative_uncertainty_num = 0
    relative_uncertainty_kep = 0
    for i in range(len(periodetidene)):
        relative_uncertainty_num += np.abs(periodetidene[i] - period_newton[i])/ period_newton[i]
        relative_uncertainty_kep += np.abs(period_kepler[i] - period_newton[i])/ period_newton[i]

    print(f"Total relativ usikkerhet - numerisk periodetid mot newtons periodetid: {relative_uncertainty_num*100} %")
    print(f"Total relativ usikkerhet - keplers periodetid mot newtons periodetid: {relative_uncertainty_kep*100} %")


if __name__ == "__main__":
    plot_information()
    planet_positions, simulation_duration, times = get_information()
    filename = "exact_trajectories.npz"
    system.verify_planet_positions(simulation_duration, planet_positions, 
                                   filename, number_of_output_points=None)
    print(times)
    system.generate_orbit_video(times, planet_positions)




# Arealet som er sveipet ut er rimelige størrelser i forhold til jorda
    

