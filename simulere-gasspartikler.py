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

def create_random_positions(n: int) -> list[float]:
    """
    Funksjon som lager en array med n posisjons-vektorer [x, y, z].
    Komponentene blir tilfeldig valgt med en uniform fordeling innenfor
    'boksen' vår.

    Parametere:
    n (int): bestemmer formen til arrayen 
    
    Return:
    (list): array med uniformt fordelte posisjoner i x-, y- og z-retning
    """
    return np.random.uniform(x1, x2, (n, 3))


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
    """
    global t, F, T_tot, dt, posisjoner, hastigheter, L, pz
    while t < T_tot:
        kollisjon_topp = posisjoner > L
        kollisjon_bunn = posisjoner < 0
        kollisjon_bunn[:, 2] = False
        

        partikkel_rømming = posisjoner[:, 2] < 0
        norm = summerer_fart(hastigheter, partikkel_rømming)

        pz += 2*m*norm

        hastigheter[kollisjon_topp | kollisjon_bunn] *= -1

        posisjoner += hastigheter*dt

        posisjoner[partikkel_rømming] = påfyll()
        hastigheter[partikkel_rømming] = create_random_velocities(1)

        t += dt
    F = pz / T_tot

    return F


F_1 = kraft()*antall
print(f"Total kraft er {F_1}, og areal {L**2 * antall}")




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
    v = np.sqrt(hastigheter[:,0]**2 + hastigheter[:,1]**2 + hastigheter[:, 2]**2)
    s = v**2 * P_v_abs(v)
    v_mid = sum(s)/len(v)
    return v_mid

def v_middel_numerisk():
    v1 = np.sqrt(hastigheter[:,0]**2 + hastigheter[:,1]**2 + hastigheter[:, 2]**2)
    v2 = sum(v1)/len(v1)
    return v2

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
    print(f"Kraften er {F} og arealet er {A}")
    pressure = F/A
    return pressure


def plot_hastighet_absolutt(hastigheter):
    """
    Plotter analytisk absolutt hastighet mot numerisk hastighet. 
    """
    v_abs = np.sqrt(hastigheter[:,0]**2 + hastigheter[:,1]**2 + hastigheter[:, 2]**2)
    v_abs_analytical = np.arange(min(v_abs)-1000, max(v_abs)+1000, 30)

    plt.plot(v_abs_analytical, P_v_abs(v_abs_analytical), label="Analytisk normalkurve")
    plt.hist(v_abs, bins=10, density=True, alpha=0.3, edgecolor="black", label='numerisk farter')
    plt.xlabel("Hastighet [m/s]")
    plt.ylabel("Sannsynlighet")
    plt.title("Analytiske mot numeriske absolutte hastigheter")
    plt.legend()
    plt.show()


def plot_hastighet_komponent(hastigheter):
    """
    Plotter analytisk hastighetskomponent mot numerisk hastighet. 
    """
    vx = hastigheter[:,0]
    v_x = np.arange(min(vx)-1000, max(vx)+1000, 30)

    plt.plot(v_x, P_v_vec(v_x), label="Analytisk normalkurve")
    plt.hist(vx, bins=10, density=True, alpha=0.3, edgecolor="black", label='numerisk farter')
    plt.xlabel("Hastighet [m/s]")
    plt.ylabel("Sannsynlighet")
    plt.title("Analytiske mot numeriske hastigheter")
    plt.legend()
    plt.show()

    
plot_hastighet_komponent(hastigheter)
plot_hastighet_absolutt(hastigheter)

P = trykk_analytisk()
p = trykk_numerisk()
print(f"Trykket er {P:.2f} Pa analytisk og på {p} Pa numerisk forholdet blir da {p/P} %")

E = energi_analytical()
e = energi_numerical()  
print(f"Energien er {E} J analytisk og på {e} J numerisk, forholdet blir da {e/E} %")

v = v_middel_numerisk()
V = v_middel_analytisk()

print(f"Farta er {v} m/s numerisk og {V} m/s analytisk, forholdet blir da {v/V} %")



# Trykke: Regne ut kreftene
# dp/dt: 
