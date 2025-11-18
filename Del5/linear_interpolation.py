# Ikke brukt kodemal
import numpy as np

def linear_interpolation(r:np.ndarray, t:float, dt:float):
    """This function does a linear interpolation such that we can estimate the 
    planet position in a time between the timesteps.
    
    Params:
    r (ndarray): array with stored positions of object
    t (float): time at which we want to find the objects position
    dt (float): timestep

    Returns:
    r_ (ndarray): position found by doing linear interpolation
    """
    rest = (t % dt) / dt
    
    indeks_start = int(t / dt)  # int() always round down
    indeks_end = indeks_start + 1

    # if the timestep that is used is different from the timestep in r
    if indeks_start > len(r) - 2:
        indeks_start = -2
        indeks_end = -1

    add_position = (r[indeks_end] - r[indeks_start]) * rest
    r_ = r[indeks_start] + add_position    

    return r_