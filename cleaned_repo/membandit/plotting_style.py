"""Matplotlib / seaborn style defaults for MemoryBandit figures."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def set_plot_style():
    # General figure settings
    plt.rcParams.update({
        'figure.dpi': 300,
        'figure.figsize': (4, 4),
        'lines.linewidth': 1.5,
        'lines.dash_capstyle': 'round',
        'font.family': 'Arial',
        'font.size': 10,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'legend.frameon': False
    })

    # Define RGB values for your two colors
    ignacio_purple = np.array([86, 28, 70]) / 255
    ignacio_blue = np.array([0, 70, 126]) / 255 

    # Create a Seaborn palette with your custom colors normalized to 0-1
    custom_palette = sns.color_palette([ignacio_purple, ignacio_blue])

    # ignacio_orange = np.array([255, 172, 77]) / 255
    # ignacio_mauve = np.array([193, 155, 211]) / 255
    # ignacio_beige = np.array([205, 182, 162]) / 255

    # Set the custom palette
    sns.set_palette(custom_palette)
    
    # from YBA
    # hpc_c = np.array([0.44314, 0.2902, 1])
    # amy_c = np.array([0.49412, 0, 0.90196])
    # ofc_c = np.array([1, 0.75686, 0.41569])
    # acc_c = np.array([0.7098, 0.27059 ,0.68235])
    # dmpfc_c = np.array([0.40784, 0.01961, 0])
    # dlpfc_c = np.array([1, 0.06275, 0.06275])

    # Seaborn style and context
    # sns.set_style("whitegrid")
    # sns.set_palette("colorblind")
    sns.set_context("paper", font_scale=1.5)

    # Remove top and right spines by default
    sns.despine()
