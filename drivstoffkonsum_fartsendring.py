# Oppgave D
# Vi har ikke bruk kodemal


F_thrust = 60000    # N
Fuel_consumption = 6.0      # kg/s
current_rocket_mass = 10000  # kg
delta_v = 7200      # m / s


def akselerasjon(m: float, F: float):
    """Regner ut akselerasjonen til raketten ved bruk av Newtons 2. lov.
    
    Parametere:
    m (float): rakettens nåværende masse
    F (float): skyvkraft til motoren

    Returnerer:
    a (float): rakettens akselerasjon
    """
    a = F / m
    return a


def fuel_consume(F:float, dv:float, m:float, dm:float):
    """Regner ut rakettens totale drivstofforbruk etter en økning i fart.
    
    Parametere:
    F (float): rakettens skyvkraft i N
    dv (float): økningen i fart vi ønsker å regne drivstofforbruket til i m/s
    m (float): rakettens masse før akselerasjonen i kg
    dm (float): rakettens drivstofforbruk i kg

    Returnerer:
    fuel_consumed (float): totalt drivstoff brukt på fartsøkningen i kg
    t (float): total tid brukt på fartsøkningen i sekunder
    final_rocket_mass (float): rakettens masse etter fartsøkningen i kg
    """
    v = 0   # løpende fart [m/s]
    t = 0   # løpende tid [s]
    dt = 0.01   # tidssteg [s]
    fuel_consumed = 0   
    final_rocket_mass = m

    while v < dv:
        a = akselerasjon(m, F)
        v += a*dt
        t += dt
    
    fuel_consumed += t * dm     # kg = s * kg/s
    final_rocket_mass -= fuel_consumed

    return fuel_consumed, t, final_rocket_mass



Fuel_consumed, t, final_rocket_mass = fuel_consume(F_thrust, delta_v, current_rocket_mass, Fuel_consumption)

print(f"Vi har en motorkraft på {F_thrust} N og drivstofforbruk på {Fuel_consumption} kg/s.")
print(f"Raketten har en startmasse på {current_rocket_mass} kg der drivstoff er {current_rocket_mass - 1100} kg")
print(f"På tiden {t:.2f} sekunder har vi oppnåd en fartsendring på {delta_v} m/s.")
print(f"Raketten brukte {Fuel_consumed:.2f} kg drivstoff på denne akselerasjonen.")
print(f"Raketten veier nå {final_rocket_mass} kg.")
