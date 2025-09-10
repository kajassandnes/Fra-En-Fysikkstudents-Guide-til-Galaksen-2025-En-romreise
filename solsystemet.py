import ast2000tools.utils as utils
import ast2000tools.constants as const
seed = utils.get_seed('oafincke')
import numpy as np


from ast2000tools.solar_system import SolarSystem
system = SolarSystem(seed)
from ast2000tools.space_mission import SpaceMission
mission = SpaceMission(seed)

print('My system has a {:g} solar mass star with a radius of {:g} kilometers.'
      .format(system.star_mass, system.star_radius))

for planet_idx in range(system.number_of_planets):
    print('Planet {:d} is a {} planet with a semi-major axis of {:g} AU.'
          .format(planet_idx, system.types[planet_idx], system.semi_major_axes[planet_idx]))



print(system.masses*const.m_sun)
print(system.masses)
print(system.radii)
print(system.number_of_planets)

M_0 = system.masses[0]*const.m_sun
G = const.G
R_0 = system.radii[0]*1e3
g_0 = G*M_0/(R_0**2)

R_1 = system.radii[1]*1e3
M_1 = system.masses[1]*const.m_sun
g_1 = G*M_1/(R_1**2)


rotasjon_dager = system.rotational_periods
rotasjon_sekunder = utils.day_to_s(rotasjon_dager[0])
vinkelfart = 2*np.pi/rotasjon_sekunder
rotasjons_fart = vinkelfart * R_0

print()
print(f"Initial fuel mass is {mission.initial_fuel_mass}")
print(f"Planeten vår bruker {rotasjon_dager[0]} dager på å spinne om sin egen akse")
print(f"På overflaten har vi en fart på {rotasjons_fart} m/s om jordaksen")
print()

print(f"Tyngdeakselerasjonen til planeten vår er {g_0} kgm/s².")
print(f"Tyngdeakselerasjonen til naboplaneten vår er {g_1} kgm/s²")
print()


v_escape = np.sqrt(2*G*M_0/R_0)
v_initial = rotasjons_fart
v_rest = np.sqrt(v_escape**2 - v_initial**2)

m = 1100 + 5000

tid = 60*20

a = v_rest/tid

print(f"Unnslipningsfarten er {v_escape} m/s, men vi har en initialfart fra jordrotasjonen på {v_initial} m/s")
print(f"Vi må oppnå en fart på {v_rest} m/s.")
print()

print(f"Vi oppnår unnslipningshastigheten på {tid} sekunder dersom vi har en akselerasjon på {a} m/s²")

F = m*a

print()
print(f"Krafta vi trenger er {F} dersom raketten veier {m} kg")

print()
print('My spacecraft has a mass of {:g} kg and a cross-sectional area of {:g} m^2.'
      .format(mission.spacecraft_mass, mission.spacecraft_area))
print()




initial_positions = system.initial_positions
initial_velocities = system.initial_velocities

print("Dette er startposisjonene til planeten vår")
print(f"({initial_positions[0][0]}, {initial_positions[1][0]})")
print()

print("Dette er startfartene til planeten vår")
print(f"({initial_velocities[0][0]}, {initial_velocities[1][0]})")
print()
