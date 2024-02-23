import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import seaborn as sns

# # Define your custom color cycle
# rgb_values = np.array([
#     ( 39, 125, 161),
#     (248, 150,  30),
#     ( 67, 170, 139),
#     (231,  98,  84),
#     ( 92,  55,  76),
#     (118, 129, 142),
#     (109, 211, 206),
#     (  0,  42,  50),
#     (125, 128, 218),
#     ( 86,  53,  30),
#     ]) / 255
# rgb_values = np.concatenate((rgb_values, np.ones((len(rgb_values), 1))), axis=1)

# rgb_values_cmap = np.array([
#     (231,  98,  84), (239, 138,  71), 
#     (247, 170,  88), (255, 208, 111),
#     (255, 230, 183), (170, 220, 224), 
#     (114, 188, 213), ( 82, 143, 173),
#     ( 55, 103, 149), ( 30,  70, 110)])[::-1] / 255

# Set the custom color cycle using rcParams
def config_matplotlib():
    
    plt.style.use('seaborn-deep')
    
    # plt.rc('font', family='Dejavu Sans')
    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    # plt.rcParams['axes.prop_cycle'] = plt.cycler(color=rgb_values)
    
    # Define a custom colormap
    # custom_cmap_name = "custom_cmap"
    # custom_cmap = LinearSegmentedColormap.from_list(custom_cmap_name, rgb_values_cmap)
    cmap = sns.color_palette("ch:start=.5,rot=-.5", as_cmap=True)
    
    # Register the colormap with matplotlib
    plt.register_cmap(cmap=cmap)
    print(f'\nregistered cmap: {cmap.name}\n')

    SMALL_SIZE = 10
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 14

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

