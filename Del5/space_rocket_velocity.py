# Ikke brukt kodemal
import ast2000tools.utils as utils
import ast2000tools.constants as const
seed = utils.get_seed('oafincke')
import numpy as np
import matplotlib.pyplot as plt

from ast2000tools.solar_system import SolarSystem
system = SolarSystem(seed)
from ast2000tools.space_mission import SpaceMission
mission = SpaceMission(seed)

c = const.c     # m/s

def radial_velocity_star_relative_to_reference_star(index: int) -> float:
    """Function that converts measured wavelengths to radial velocity. 
    
    Parameters: 
    index (int): number at star you want to find the dopplershift for. There 
                 is two stars.
    dlambda (float): difference in measured wavelength and measured wavelength 
                     at rest-frame (nanometers)
    lambda0 (float): measured wavelength of H_alpha at rest-frame (nanometers)
    
    Return:
    v_r (float): relative radial velocity between star and reference star (m/s)
    """
    lambda0 = mission.reference_wavelength  # nanometers
    dlambda = mission.star_doppler_shifts_at_sun[index] # Doppler shifts in nano meters


    global c
    v_r = c*dlambda / lambda0   # m/s

    return v_r


def radial_velocity_spacecraft_relative_to_reference_star(dlambda:float) -> float:
    """Finner radiell hastighet mellom rakett og referansestjerne.

    Parameter:
    dlambda (float): dopplerskift til ønsket referansestjerne i nanometer

    Returnerer:
    v_r (float): radiell fart til raketten relativt til referansestjerne (m/s)
    """
    lambda0 = mission.reference_wavelength  # nanometers

    global c    # m/s
    v_r = c*dlambda / lambda0   # m/s

    return v_r


def vinkel_mellom_referansestjerne_og_x_akse(index: int) -> float:
    """Henter ut vinkel mellom valgt referansestjerne og x-aksen med ast2000tools
    og konverterer til radianer.

    Parameter:
    index (int): index til referansestjerne - index = 0 -> stjerne 1
                                            - index = 1 -> stjerne 2
    Returnerer:
    rad (float): vinkel mellom x-akse og referansestjerne i radianer
    """
    phi = mission.star_direction_angles[index] # degrees
    rad = utils.deg_to_rad(phi)  # radian

    return rad

def radiell_til_kartesisk(radiell:np.ndarray) -> np.ndarray:
    """Funksjon som transformerer farter fra rommet der de radielle enhetsvektorene 
    danner en basis til rommet der de kartesiske enhetsvektorene danner en basis.
    
    Dette gjøres via en matriseligning:
        vx = v1*cos(phi1) + v2*cos(phi2)
        vy = v1*sin(phi1) + v2*sin(phi2)
    
    Parameter:
    radiell (ndarray): array med romskipets eller stjernas radielle farter 
                     relativt til referansestjernene (m/s)

    Returnerer:
    v (ndarray): array med romskipets eller stjernas kartesiske farter relativt 
               til referansestjernene (m/s)
    """
    phi_1 = vinkel_mellom_referansestjerne_og_x_akse(0)   # radians
    phi_2 = vinkel_mellom_referansestjerne_og_x_akse(1)   # radians

    v_x = np.cos(phi_1)*radiell[0] + np.cos(phi_2)*radiell[1]   # m/s
    v_y = np.sin(phi_1)*radiell[0] + np.sin(phi_2)*radiell[1]   # m/s

    v = np.array([v_x, v_y])    # m/s
    
    return v



def kartesisk_fart_rakett(dlambda1:float, dlambda2:float) -> np.ndarray:
    """Funksjon som finner farta til raketten vår i kartesiske koordinater.
    
    Parametere:
    dlamda1 (float): dopplerskiftet til romskipet måler hos referansestjerne 1 (nanometer)
    dlamda2 (float): dopplerskiftet til romskipet måler hos referansestjerne 2 (nanometer)

    Returnerer:
    v_xy_spaceship_relative_to_star (ndarray): romskipets kartesiske farter 
                                               relativt til stjerne.
    """
    # finner fartene til romskipet radielt til referansestjernene
    v_r1_spaceship = radial_velocity_spacecraft_relative_to_reference_star(dlambda1)
    v_r2_spaceship = radial_velocity_spacecraft_relative_to_reference_star(dlambda2)
    # Setter komponentene i en array
    v_r_spaceship = np.array([v_r1_spaceship, v_r2_spaceship])

    # finner fartene til stjerna radielt til referansestjernene
    v_r1_star = radial_velocity_star_relative_to_reference_star(0) # m/s
    v_r2_star = radial_velocity_star_relative_to_reference_star(1) # m/s
    # Setter komponentene i en array
    v_r_star = np.array([v_r1_star, v_r2_star])

    # transformerer koordinatene til romskipet fra radielle til kartesiske
    v_xy_spaceship = radiell_til_kartesisk(v_r_spaceship) # m/s

    # transformerer koordinatene til stjerna fra radielle til kartesiske
    v_xy_star = radiell_til_kartesisk(v_r_star) # m/s

    v_xy_spaceship_relative_to_star = v_xy_spaceship - v_xy_star

    return v_xy_spaceship_relative_to_star


def print_all_information() -> str | float:
    """Funksjon som printer informasjon om systemet vårt: 
        - stjernas radielle fart i forhold til referansestjerer (m/s)
        - vinkel mellom x-aksen og stjernene (radianer)
        - stjernas kartesiske farter i forhold til referansestjerner (m/s)
        - romskipets kartesiske farter i forhold til Frogstar (m/s)
    """
    v_r1_star = radial_velocity_star_relative_to_reference_star(0) # m/s
    v_r2_star = radial_velocity_star_relative_to_reference_star(1) # m/s
    print(f"Radial velocity Frogstar, star 1: {v_r1_star:.2f} m/s")
    print(f"Radial velocity Frogstar, star 2: {v_r2_star:.2f} m/s")
    print()

    phi_1 = vinkel_mellom_referansestjerne_og_x_akse(0)   # radians
    phi_2 = vinkel_mellom_referansestjerne_og_x_akse(1)   # radians
    print(f"Angle between x-axis and star 1: {phi_1:.2f} radians")
    print(f"Angle between x-axis and star 2: {phi_2:.2f} radians")
    print()

    v_r_star = np.array([v_r1_star, v_r2_star])
    v_xy_star = radiell_til_kartesisk(v_r_star)
    print(f"Cartesian coordinates of spaceship relative to reference stars (m/s): ({v_xy_star[0]}, {v_xy_star[1]})")
    print()

    v_xy_spaceship_relative_to_star = kartesisk_fart_rakett(0, 0)
    print(f"Fart til romskip i kartesiske koordinater relativt til Frogstar (m/s): {v_xy_spaceship_relative_to_star}")
    print()






if __name__ == "__main__":
    print_all_information()