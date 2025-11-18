# Ikke brukt kodemal
import ast2000tools.utils as utils
seed = utils.get_seed('oafincke')
import numpy as np
from ast2000tools.solar_system import SolarSystem
system = SolarSystem(seed)
from ast2000tools.space_mission import SpaceMission
mission = SpaceMission(seed)
from PIL import Image

def make_sample_picture():
    """The function recreates the picture 'sample0000.png' to make shure that
    orientation is correct implemented.
    """
    img = Image.open('sample0000.png') # Open existing png

    pixels = np.array(img) # png into numpy array
    length = len(pixels)
    width = len(pixels[0, :])
    # Known values for Field of View (alpha) and position centered (phi0, theta0)
    alpha_phi = utils.deg_to_rad(70)
    alpha_theta = utils.deg_to_rad(70)
    phi0 = utils.deg_to_rad(0)
    theta0 = utils.deg_to_rad(90)   # spaceship plane
    # using formulas for x_max and y_max
    x_max = (2 * np.sin(alpha_phi/2)) / (1 + np.cos(alpha_phi/2))
    y_max = (2 * np.sin(alpha_theta/2)) / (1 + np.cos(alpha_theta/2))
    # making a meshgrid for pixels in picture
    X_range = np.linspace(-x_max, x_max, width)
    Y_range = np.linspace(y_max, -y_max, length)
    X, Y = np.meshgrid(X_range, Y_range)
    # angles corresponding to X- and Y-coordinates 
    rho = np.sqrt(X**2 + Y**2)
    beta = 2 * np.arctan(rho/2)
    theta_range = theta0 - np.arcsin(np.cos(beta) * np.cos(theta0) + Y/rho * np.sin(beta) * np.sin(theta0))
    phi_range = phi0 + np.arctan(X * np.sin(beta) / (rho * np.sin(theta0) * np.cos(beta) - Y*np.cos(theta0) * np.sin(beta)))

    index_pixels = np.zeros((length, width))
    # finding pixel corresponding to angle-coordinates
    for i in range(length):
        for j in range(width):
            index_pixels[i][j] = mission.get_sky_image_pixel(theta_range[i][j], phi_range[i][j])
    # finding colors corresponding to each pixle
    himmelkule = np.load("himmelkule.npy")
    new_pixels = himmelkule[index_pixels.astype(int), 2:5].astype(np.uint8)
    # making new picture
    img2 = Image.fromarray(new_pixels)
    img2.save('sample0000new.png') # Make new png


def make_and_save_360_pictures():
    """Goes through all 360 degrees for phi to make reference pictures
    which the spaceship will use to orient itself.
    """
    img = Image.open('sample0000.png') # Open existing png
    # setting same parameters as in 'make_sample_picture()'       
    pixels = np.array(img) # png into numpy array
    length = len(pixels)
    width = len(pixels[0, :])

    alpha_phi = utils.deg_to_rad(70)
    alpha_theta = utils.deg_to_rad(70)
    theta0 = utils.deg_to_rad(90)

    x_max = (2 * np.sin(alpha_phi/2)) / (1 + np.cos(alpha_phi/2))
    y_max = (2 * np.sin(alpha_theta/2)) / (1 + np.cos(alpha_theta/2))

    X_range = np.linspace(-x_max, x_max, width)
    Y_range = np.linspace(y_max, -y_max, length)

    X, Y = np.meshgrid(X_range, Y_range)

    rho = np.sqrt(X**2 + Y**2)
    beta = 2 * np.arctan(rho/2)
    theta_range = theta0 - np.arcsin(np.cos(beta) * np.cos(theta0) + Y/rho * np.sin(beta) * np.sin(theta0))
    # Doing the same as in 'make_sample_picture()' 
    all_pics = []
    for i in range(360):
        phi0 = utils.deg_to_rad(i)
        phi_range = phi0 + np.arctan(X * np.sin(beta) / (rho * np.sin(theta0) * np.cos(beta) - Y*np.cos(theta0) * np.sin(beta)))

        index_pixels = np.zeros((length, width))

        for i in range(length):
            for j in range(width):
                index_pixels[i][j] = mission.get_sky_image_pixel(theta_range[i][j], phi_range[i][j])

        himmelkule = np.load("himmelkule.npy")
        new_pixels = himmelkule[index_pixels.astype(int), 2:5].astype(np.uint8)
        # saving all pictures in a list
        all_pics.append(new_pixels)
    # Making the list into an array with all the list-elements
    all_pics = np.stack(all_pics)
    np.savez("360_reference_pictures.npz", all_pics=all_pics)


def image_analysis(picture_png):
    """Compare the RGB colours for each pixel and find the phi-value for the 
    picture with least deviation from comparing picture.
    
    Params:
    picture_png (str): filename of .png picture

    Returns:
    phi (float): angle in the azimuthal plane (radians)
    """
    img = Image.open(picture_png)

    data = np.load("360_reference_pictures.npz")['all_pics']
    pixels = np.array(img).astype(np.float32) # png into numpy array
    # Finding the differences
    pictures = data.astype(np.float32)
    difference = np.linalg.norm(pixels - pictures, axis=3)
    deviations = np.sum(difference, axis=(1,2))
    phi = np.argmin(deviations) # degrees

    return phi
        

if __name__ == "__main__":
    #make_sample_picture()
    #make_and_save_360_pictures()
    phi = image_analysis('sample0000.png')
    print(phi)

    