# Oppgave D
# Vi har ikke bruk kodemal
import numpy as np
from ast2000tools import utils

def akselerasjon(m: float, F: float):
    """Regner ut akselerasjonen til raketten ved bruk av Newtons 2. lov.
    
    Parametere:
    m (float): rakettens nåværende masse (kg)
    F (float): skyvkraft til motoren (N)

    Returnerer:
    a (float): rakettens akselerasjon (m/s²)
    """
    a = F / m
    return a


def fuel_consume(F:float, dv:float, m:float, dm:float):
    """Regner ut rakettens totale drivstofforbruk etter en økning i fart.
    
    Parametere:
    F (float): motorens skyvkraft (N)
    dv (float): fartsøkning (m/s)
    m (float): rakettens masse før akselerasjonen (kg)
    dm (float): rakettens drivstofforbruk (kg/s)

    Returnerer:
    fuel_consumed (float): totalt drivstoff brukt på fartsøkningen (kg)
    time_of_boost (float): total tid brukt på fartsøkningen (s)
    final_rocket_mass (float): rakettens masse etter fartsøkningen (kg)
    """
    v = 0  # m/s
    time_of_boost = 0   # s
    dt = 0.01   # tidssteg [s]
    fuel_consumed = 0   
    final_rocket_mass = m

    while v < dv:
        a = akselerasjon(m, F)
        v += a*dt
        time_of_boost += dt
    
    fuel_consumed += time_of_boost * dm     # kg = s * kg/s
    final_rocket_mass -= fuel_consumed

    return fuel_consumed, time_of_boost, final_rocket_mass


def rocket_properties():
    """Give access to the property of the rocket."""
    thrust = 60000  # N
    mass_loss_rate = 6.0    # kg/s
    initial_fuel_mass = 6500    # kg

    return thrust, mass_loss_rate, initial_fuel_mass


if __name__ == "__main__":
    thrust, mass_loss_rate, initial_fuel_mass = rocket_properties()
    current_rocket_mass = 1904.68 # kg
    delta_v = 9615      # m/s
    delta_v = np.linalg.norm(np.array([2650, 2100]))

    fuel_consumed, time_of_boost, final_rocket_mass = fuel_consume(thrust, delta_v, current_rocket_mass, mass_loss_rate)

    print(f"Vi har en motorkraft på {thrust} N og drivstofforbruk på {mass_loss_rate} kg/s.")
    print(f"Raketten har en masse på {current_rocket_mass} kg etter launch, der drivstoff er {current_rocket_mass - 1100} kg")
    print(f"På tiden {time_of_boost:.2f} sekunder har vi oppnåd en fartsendring på {delta_v} m/s.")
    print(f"Raketten brukte {fuel_consumed:.2f} kg drivstoff på denne akselerasjonen.")
    print(f"Raketten veier nå {final_rocket_mass} kg.")
    print(f"Det er nå {final_rocket_mass - 1100} kg drivstoff igjen")
    print(f"Heldigvis fylte vi på tanken etter at vi launcha.")