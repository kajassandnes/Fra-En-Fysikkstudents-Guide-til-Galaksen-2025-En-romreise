# This code simulate the gassparticles in an engine
import ast2000tools.constants as const
import numpy as np
import random as rd
import matplotlib.pyplot as plt


# ========== Konstanter ======================================
N = 100000           # number off particles in engine
L = 1e-6         # length of box (m)
T = 3e3          # temperature (K)
k = const.k_B      # Boltzmann-konstanten
x1, x2 = 0, L     # start og slutt (m)
m = const.m_H2  # masse til hydrogenmolekyl (kg)
mean = 0        # mu
sigma = np.sqrt(k*T/m)  # standardavvik
# ============================================================
rd.seed(100)




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


def påfyll():
    """
    Antall partikler i motoren skal være konstant. Når partikler unnslipper
    boksen, fyller vi på nye. De skal fylles på øverst i boksen, altså z = L, 
    men vi legger på en margin på ett tidssteg, slik at partiklene holder seg 
    i boksen selv om de har fart oppover.

    Return:
    (list): array med uniformt fordelte posisjoner i x-, y- og z-retning
    """
    x = np.random.uniform(x1, x2)
    y = np.random.uniform(x1, x2)
    posisjon = np.array([x, y, L-dt])
    return posisjon

posisjoner = create_random_positions(N) 




# Second step: Generate random velocities in each directions for all the particles

def create_random_velocities(n: int) -> list[float]:
    """
    Funksjon som lager en array med N hastighets-vektorer [vx, vy, vz].
    Komponentene blir tilfeldig valgt med en gaussisk fordeling.

    Parametere:
    n (int): bestemmer formen til arrayen 
    
    Return:
    (list): array med gauss-fordelte hastigheter i x-, y- og z-retning

    """
    return np.random.normal(mean, sigma, (n, 3))

hastigheter = create_random_velocities(N)



def summerer_fart(hastigheter: list[float], partikkel_rømming: list[True | False]) -> float:
    """
    Summerer opp z-komponentene til hastighetene til alle partiklene som har 
    sluppe ut under boksen. Disse brukes til å finne drivet og krafta senere.

    Parametere:
    hastigheter (list[n, 3]): array med hastighetsvektorene til alle partiklene
    partikkel_rømming (list): array med True der partikkelen har kommet ned og
    ut av boksen, False hvis partikkelen er inni boksen

    Return:
    (float): summen av alle z-komponentene. Den totale "hastigheten" på vei ut
    rett nedover.
    """
    norm = sum(hastigheter[partikkel_rømming])

    if isinstance(norm, np.ndarray):
        z_norm = abs(norm[2])
        return z_norm
    else:
        return 0


# Third step: Follow the movements of the gass particles with time
# =======================================
t = 0
dt = 1e-12
T_tot = 1e-9
F = 0
antall = 7e12
pz = 0
# ========================================

def kraft(posisjoner, hastigheter):
    """
    Regner ut kraften partiklene i én enkelt boks med lengde 10⁻⁶ produserer
    og ganger det med 2.5*10¹². Da får motoren et areal på 2.5 m². 

    Parametere:
    t, dt, T_tot (float): starttid, tidssteg, total tid
    F (float): kraften
    posisjoner, hastigheter (list(n, 3)): sier seg selv
    L (float): lengde til boksen
    antall (int): antall bokser vi skal gange krafta med
    """
    global t, F, T_tot, dt, L, pz
    while t < T_tot:
        kollisjon_topp = posisjoner > L
        kollisjon_bunn = posisjoner < 0
        kollisjon_bunn[:, 2] = False
        

        partikkel_rømming = posisjoner[:, 2] < 0
        norm = summerer_fart(hastigheter, partikkel_rømming)

        pz += 2*m*norm

        hastigheter[kollisjon_topp | kollisjon_bunn] *= -1  # partikler som kolliderer skifter vei

        posisjoner += hastigheter*dt

        posisjoner[partikkel_rømming] = påfyll()
        hastigheter[partikkel_rømming] = create_random_velocities(1)

        t += dt

    F = pz / T_tot

    return F

# Regn ut gjennomsnitt og standardavvik til krafta
def mean_og_sd_kraft():
    global antall, posisjoner, hastigheter

    gang = 1000
    liste = np.zeros(gang)
    for i in range(gang):
        liste[i] = kraft(posisjoner, hastigheter)*antall
        posisjoner = create_random_positions(N) 
        hastigheter = create_random_velocities(N)

    print(liste)

    gjennomsnitt = np.mean(liste)
    standardavvik = np.std(liste)
    print(f"mean = {gjennomsnitt}, SD = {standardavvik}")

    



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
    global hastigheter
    v_abs_numerisk = np.sqrt(hastigheter[:,0]**2 + hastigheter[:,1]**2 + hastigheter[:, 2]**2)
    v_kvadrert = v_abs_numerisk**2
    #s = v_kvadrert * P_v_abs(v_abs_numerisk)
    v_mid_kvadrert = sum(v_kvadrert)/len(v_abs_numerisk)
    return v_mid_kvadrert

def v_middel_numerisk():
    v_abs = np.sqrt(hastigheter[:,0]**2 + hastigheter[:,1]**2 + hastigheter[:, 2]**2)
    v_middel = sum(v_abs)/len(v_abs)
    return v_middel

def v_middel_analytisk():
    return 2 * np.pi**(-1/2) * np.sqrt((2*k*T) / m)

def energi_analytical():
    E = 3/2*k*T
    return E
    
def energi_numerical():
    v_mid_kvadrert = v_kvadrert_middel_numerisk()
    E = 1/2 * m * v_mid_kvadrert
    return E

def trykk_analytisk(): 
    global k, T, N, L 
    n = N/(L**3)  
    return n*k*T

def trykk_numerisk():
    global L, F
    A = L**2
    #print(f"Kraften er {F} og arealet er {A}")
    pressure = F/A
    return pressure



def plot_hastighet_absolutt(hastigheter):
    """
    Plotter analytisk absolutt hastighet mot numerisk hastighet. 
    """
    v_abs_numerisk = np.sqrt(hastigheter[:,0]**2 + hastigheter[:,1]**2 + hastigheter[:, 2]**2)
    v_abs_analytical = np.arange(min(v_abs_numerisk)-1000, max(v_abs_numerisk)+1000, 30)

    plt.plot(v_abs_analytical, P_v_abs(v_abs_analytical), label="Analytisk normalkurve")
    plt.hist(v_abs_numerisk, bins=10, density=True, alpha=0.3, edgecolor="black", label='numerisk farter')
    plt.xlabel("Hastighet [m/s]")
    plt.ylabel("Sannsynlighet")
    plt.title("Analytiske mot numeriske absolutte hastigheter")
    plt.legend()
    plt.show()


def plot_hastighet_komponent(hastigheter):
    """
    Plotter analytisk hastighetskomponent mot numerisk hastighet. 
    """
    v_x_numerisk = hastigheter[:,0]
    v_x_analytisk = np.arange(min(v_x_numerisk)-1000, max(v_x_numerisk)+1000, 30)

    plt.plot(v_x_analytisk, P_v_vec(v_x_analytisk), label="Analytisk normalkurve")
    plt.hist(v_x_numerisk, bins=10, density=True, alpha=0.3, edgecolor="black", label='numerisk farter')
    plt.xlabel("Hastighet [m/s]")
    plt.ylabel("Sannsynlighet")
    plt.title("Analytiske mot numeriske hastigheter")
    plt.legend()
    plt.show()




# Kraft
F_1 = kraft(posisjoner, hastigheter)*antall

# Trykk
p_analytisk = trykk_analytisk()
p_numerisk = trykk_numerisk()

# Energi
E_analytisk = energi_analytical()
E_numerisk = energi_numerical()  

# Fart
v_numerisk = v_middel_numerisk()
v_analytisk = v_middel_analytisk()


if __name__ == "__main__":
    # Plott
    plot_hastighet_komponent(hastigheter)
    plot_hastighet_absolutt(hastigheter)

    # Kraft
    print(f"Motorens skyvkraft er {F_1}, og areal på motoren er {L**2 * antall}")
    print()

    # Trykk
    print(f"Trykket er {p_analytisk} Pa analytisk og på {p_numerisk}")
    print(f"Pa numerisk forholdet blir da {abs(p_numerisk - p_analytisk)/p_analytisk * 100} %")
    print()

    # energi
    print(f"Energien er {E_analytisk} J analytisk og på {E_numerisk} J numerisk")
    print(f"forholdet blir da {abs(E_numerisk - E_analytisk)/E_analytisk * 100} %")
    print()

    # Fart
    print(f"Farta er {v_analytisk} m/s analytisk og {v_numerisk} m/s numerisk")
    print(f"forholdet blir da {abs(v_numerisk-v_analytisk)/v_analytisk * 100} %")
    print()

    #mean_og_sd_kraft()


# Trykke: Regne ut kreftene
# dp/dt: 
