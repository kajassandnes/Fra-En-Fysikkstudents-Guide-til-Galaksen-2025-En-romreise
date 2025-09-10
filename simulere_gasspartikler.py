# Vi har ikke brukt kodemal
# Kommentar i bunnen av koden
# This code simulate the gassparticles in an engine
import ast2000tools.constants as const
import ast2000tools.utils as utils
import numpy as np
import random as rd
import matplotlib.pyplot as plt


# ========== Konstanter ======================================
N = 100000           # number off particles in engine
L = 1e-6         # length of box (m)
T = 3.5e3          # temperature (K)
k = const.k_B      # Boltzmann-konstanten
x1, x2 = 0, L     # start og slutt (m)
m = const.m_H2  # masse til hydrogenmolekyl (kg)
mean = 0        # mu
sigma = np.sqrt(k*T/m)  # standardavvik
# ============================================================
rd.seed(101)




# First step: simulate random positions to the N particles 

def create_random_positions(N: int) -> list[float]:
    """
    Funksjon som lager en array med n posisjons-vektorer [x, y, z].
    Komponentene blir tilfeldig valgt med en uniform fordeling innenfor
    'boksen' vår.

    Parametere:
    n (int): bestemmer formen til arrayen 
    
    Return:
    (list): array med uniformt fordelte posisjoner i x-, y- og z-retning
    """
    return np.random.uniform(x1, x2, (N, 3))


def påfyll(n: int) -> list[float]:
    """
    Antall partikler i motoren skal være konstant. Når partikler unnslipper
    boksen, fyller vi på nye. De skal fylles på øverst i boksen, altså z = L, 
    men vi legger på en margin på ett tidssteg, slik at partiklene holder seg 
    i boksen selv om de har fart oppover.

    Return:
    (list): array med uniformt fordelte posisjoner i x-, y- og z-retning
    """
    x = np.random.uniform(x1, x2, n)
    y = np.random.uniform(x1, x2, n)
    z = np.ones_like(x) * (L - dt)
    return np.column_stack((x, y, z))

posisjoner = create_random_positions(N) 




# Second step: Generate random velocities in each directions for all the particles

def create_random_velocities(n: int) -> list[float]:
    """
    Funksjon som lager en array med N hastighets-vektorer [vx, vy, vz].
    Komponentene blir tilfeldig valgt med en gaussisk fordeling.

    Parametere:
    n (int): bestemmer formen til arrayen 
    
    Return:
    (list): array med uniformt fordelte posisjoner i x- og y-retning, og z = L

    """
    return np.random.normal(mean, sigma, (n, 3))

def create_random_velocities_down(n: int) -> list[float]:
    farter = np.random.normal(mean, sigma, (n, 3))
    farter[:, 2] = (-1) * np.abs(farter[:, 2])
    return farter

hastigheter = create_random_velocities(N)




# Third step: Follow the movements of the gass particles with time
# =======================================
t = 0
dt = 1e-12
T_tot = 1e-9
F = 0
antall = 7e12   
dp_z = 0
counter = 0
# ========================================

def kraft():
    """
    Regner ut kraften partiklene i én enkelt boks med lengde 10⁻⁶ produserer
    og ganger det med 2.5*10¹². Da får motoren et areal på 2.5 m². 

    Parametere:
    t, dt, T_tot (float): starttid, tidssteg, total tid
    F (float): kraften
    posisjoner, hastigheter (list(n, 3)): sier seg selv
    L (float): lengde til boksen
    antall (int): antall bokser vi skal gange krafta med

    Returnerer:
    Kraft (float) dp/dt
    """
    global t, F, T_tot, dt, L, dp_z, posisjoner, hastigheter, counter
    while t < T_tot:
        posisjoner += hastigheter * dt  # s = v*t

        kollisjon_topp = posisjoner >= L    # sjekker hvor partiklene er 
        kollisjon_bunn = posisjoner <= 0    # utafor boksen
        kollisjon_bunn[:, 2] = False        # gjør at partikler med z-komponent
                                            # under null ikke kolliderer
        
        partikkel_rømming = posisjoner[:, 2] < 0    # partikler som unnslipper
        if np.count_nonzero(partikkel_rømming):    
            counter += np.count_nonzero(partikkel_rømming)   # Teller antall rømte partikler
            norm = np.abs(np.sum((hastigheter[partikkel_rømming, 2])))  # Total fart til alle partikler som har unnsluppet

            dp_z +=  2*m*norm     # endring i driv i z-retning

            # Fyller på nye partikler
            posisjoner[partikkel_rømming] = påfyll(len(hastigheter[partikkel_rømming])) 

            

        hastigheter[kollisjon_topp | kollisjon_bunn] *= -1  # partikler som kolliderer skifter vei

        t += dt

    F = dp_z / T_tot    # F = dp/dt

    return F


def trykk():
    """Regner ut trykket numerisk i den samme boksen, men uten hull nederst."""
    global L, mean, sigma
    t = 0
    dt = 1e-12
    tid = 1e-9
    dp = 0
    N = 100000

    r = np.random.uniform(0, L, (N, 3)) 
    v = np.random.normal(mean, sigma, (N, 3))
    

    while t < tid:
        r += v*dt

        kollisjon_topp = r >= L
        kollisjon_bunn = v <= 0

        kollisjon_xy_plan = r[:, 2] <= 0

        if np.count_nonzero(kollisjon_xy_plan):
            norm = np.abs(np.sum((v[kollisjon_xy_plan, 2])))

        dp += 2*m*norm

        v[kollisjon_topp | kollisjon_bunn] *= -1

        t += dt

    F_ = dp/tid

    A_ = L**2
    trykket = F_ / A_
    return trykket
    



# Vi skal sammenligne middel-energi, trykk, og middel-hastighet
# numerisk mot analytisk
def P_v_vec(v):
    """
    Maxwell-Boltzmann-funksjonen for hastighetskomponentene. 
    """
    return (m/(2*np.pi*k*T))**(1/2) *  np.exp((-1/2)*(m*v**2)/(k*T))

def P_v_abs(v):
    """
    Maxwell-Boltzmann-funksjonen for absolutt hastighet. 
    """
    return (m/(2*np.pi*k*T))**(3/2) *  np.exp((-1/2)*(m*v**2)/(k*T)) * 4 * np.pi * v**2



def v_kvadrert_middel_numerisk():
    """Finner absoluttverdi til hastigheten til alle partiklene, kvadrerer dem,
    summerer dem og deler dem på antall partikler."""
    global hastigheter
    v_abs_numerisk = np.linalg.norm(hastigheter[:], axis=1)
    v_kvadrert = v_abs_numerisk**2
    v_mid_kvadrert = sum(v_kvadrert)/len(v_kvadrert)
    return v_mid_kvadrert

def v_middel_numerisk():
    """Finner middelfart ved å ta absoluttverdiene til hastighetskomponentene
    hver for seg, summere dem og så dele på antall partikler."""
    v_abs_numerisk = np.linalg.norm(hastigheter[:], axis=1)
    v_middel_numerisk = sum(v_abs_numerisk)/len(v_abs_numerisk)
    return v_middel_numerisk

def v_middel_analytisk():
    """Analytisk løsning for middelfart utledet i rapporten."""
    return 2 * np.pi**(-1/2) * np.sqrt((2*k*T) / m)

def energi_analytical():
    """Analytisk energi utledet i rapporten."""
    E = 3/2*k*T
    return E
    
def energi_numerical():
    """Finner numerisk energi ved å bruke funksjonen som finner numerisk
    kvadrert middelfart."""
    v_mid_kvadrert = v_kvadrert_middel_numerisk()
    E = 1/2 * m * v_mid_kvadrert
    return E

def trykk_analytisk(): 
    """Analytisk trykk som følger tilstandslikningen."""
    global k, T, N, L 
    n = N/(L**3)  
    return n*k*T

def trykk_numerisk():
    return trykk()



def plot_hastighet_absolutt(hastigheter):
    """
    Plotter analytisk absolutt hastighet mot numerisk hastighet. 
    """
    v_abs_numerisk = np.sqrt(hastigheter[:,0]**2 + hastigheter[:,1]**2 + hastigheter[:, 2]**2)
    v_abs_analytical = np.arange(0, max(v_abs_numerisk)+1000, 30)

    plt.rcParams.update({'font.size':25})
    plt.plot(v_abs_analytical, P_v_abs(v_abs_analytical), label="Analytisk normalkurve")
    plt.hist(v_abs_numerisk, bins=10, density=True, alpha=0.3, edgecolor="black", label='Numerisk farter')
    plt.xlabel("Hastighet [m/s]", fontsize=25)
    plt.ylabel("Sannsynlighet", fontsize=25)
    plt.title("Analytiske mot numeriske absolutte hastigheter", fontsize=25)
    plt.legend()
    plt.show()


def plot_hastighet_komponent(hastigheter):
    """
    Plotter analytisk hastighetskomponent mot numerisk hastighet. 
    """
    v_x_numerisk = hastigheter[:,0]
    v_x_analytisk = np.arange(min(v_x_numerisk)-1000, max(v_x_numerisk)+1000, 30)

    plt.rcParams.update({'font.size':25})
    plt.plot(v_x_analytisk, P_v_vec(v_x_analytisk), label="Analytisk normalkurve")
    plt.hist(v_x_numerisk, bins=10, density=True, alpha=0.3, edgecolor="black", label='Numerisk farter')
    plt.xlabel("Hastighet [m/s]", fontsize=25)
    plt.ylabel("Sannsynlighet", fontsize=25)
    plt.title("Analytiske mot numeriske hastigheter", fontsize=25)
    plt.legend()
    plt.show()




# Kraft
F_1 = kraft() * antall

# Trykk
p_analytisk = trykk_analytisk()
p_numerisk = trykk_numerisk()

# Energi
E_analytisk = energi_analytical()
E_numerisk = energi_numerical()  

# Fart
v_numerisk = v_middel_numerisk()
v_analytisk = v_middel_analytisk()

# Rømte partikler
n_escaped = counter * antall
n_escaped_per_second = n_escaped / T_tot
mass_per_second = n_escaped_per_second * m


if __name__ == "__main__":
    # Plott
    plot_hastighet_komponent(hastigheter)
    plot_hastighet_absolutt(hastigheter)

    # Kraft
    print("SKYVKRAFT OG MASSE")
    print(f"Motorens skyvkraft: {F_1} N   Areal på motor: {L**2 * antall} m²")
    print(f"Antall rømte partikler på {T_tot} sekunder: {n_escaped:.4f}.")
    print(f"Rømte partikler per sekund: {n_escaped_per_second} 1/s")
    print(f"Motorens forbruk: {mass_per_second} kg/s.")
    print(f"Motorens forbruk på 20 min: {mass_per_second * 1200} kg.")
    print()

    # Trykk
    print("TRYKK")
    print(f"Trykk analytisk: {p_analytisk} Pa")
    print(f"Trykk numerisk: {p_numerisk} Pa")
    print(f"Relativ usikkerhet: {abs(p_numerisk - p_analytisk)/p_analytisk * 100} %")
    print()

    # energi
    print("ENERGI")
    print(f"Middelenergi analytisk: {E_analytisk} J")
    print(f"Middelenergi numerisk {E_numerisk} J")
    print(f"Relativ usikkerhet: {abs(E_numerisk - E_analytisk)/E_analytisk * 100} %")
    print()

    # Fart
    print("FART")
    print(f"Middelfart analytisk: {v_analytisk} m/s")
    print(f"Middelfart numerisk: {v_numerisk} m/s")
    print(f"Relativ usikkerhet: {abs(v_numerisk-v_analytisk)/v_analytisk * 100} %")
    print()



# Noen ganger simuleringen kjører oppnår trykket en usikkerhet på helt opp til 
# 15 prosent. Vi antar at dette er på grunn av usikkerhet rundt implementering
# av partikkelkollisjoner slik at noen partikler unnslipper boksen uten at 
# det blir fanget opp. Det er ofte de med høyest hastighet, og vi mister mye
# trykk.