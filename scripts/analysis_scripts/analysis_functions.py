import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

import sys 
sys.path.append("/Users/claireleblanc/Documents/grad_school/staller_lab/NN_interpretability_for_AD_prediction/Model")
from ADModel_act import ADModel_act
from ADModel_abund import ADModel_abund
from ADModel_three_state import ADModel_three_state_abund
from ADModel_two_state import ADModel_two_state_abund
import re


def get_best_model(model_results, spearman_act = True, pearson_act = True, spearman_abund = False, pearson_abund = False):
    """
    Function used in all our NNs to standardize the choice of the best NN

    Parameters 
    ----------
    model_results : pandas dataframe 
        Loaded results csv from make_summary_file.py
    
    spearman_act : bool
        Whether to use validation spearman correlation for activity data to choose best model

    pearson_act : bool
        Whether to use validation pearson correlation for activity data to choose best model
    
    spearman_abund : bool
        Whether to use validation spearman correlation for abundance data to choose best model

    pearson_abund : bool
        Whether to use validation pearson correlation for abundance data to choose best model

    Returns
    --------
    pandas dataframe
        Pandas dataframe sorted based on performance (best first)

    """

    if pearson_act:
        model_results = model_results.sort_values("pearson_act_val", ascending=False)
        model_results = model_results.reset_index()
        model_results["r1"] = model_results.index
        model_results["sum_rank"] = model_results["r1"]

        model_results['max_rank'] = model_results['r1']
    
    if spearman_act: 
        model_results = model_results.sort_values("spearman_act_val", ascending=False)
        model_results = model_results.reset_index()
        model_results["r2"] = model_results.index

        if "sum_rank" in model_results.columns: 
            model_results["sum_rank"] =  model_results["sum_rank"] + model_results["r2"]
        else: 
            model_results["sum_rank"] =  model_results["r2"]

        if not ("max_rank" in model_results.columns):
            model_results["max_rank"] =  model_results["r2"]
        else: 
            model_results['max_rank'] = model_results[['max_rank', "r2"]].max(axis=1)


        if 'level_0' in model_results.columns:
            model_results = model_results.drop(columns="level_0")
    
    if pearson_abund: 
        model_results = model_results.sort_values("pearson_abund_val", ascending=False)
        model_results = model_results.reset_index()
        model_results["r3"] = model_results.index
        
        if "sum_rank" in model_results.columns: 
            model_results["sum_rank"] =  model_results["sum_rank"] + model_results["r3"]
        else: 
            model_results["sum_rank"] =  model_results["r3"]
        
        if not ("max_rank" in model_results.columns):
            model_results["max_rank"] =  model_results["r3"]
        else: 
            model_results['max_rank'] = model_results[['max_rank', "r3"]].max(axis=1)



        if 'level_0' in model_results.columns:
            model_results = model_results.drop(columns="level_0")

    if spearman_abund: 
        model_results = model_results.sort_values("spearman_abund_val", ascending=False)
        model_results = model_results.reset_index()
        model_results["r4"] = model_results.index

        if "sum_rank" in model_results.columns: 
            model_results["sum_rank"] =  model_results["sum_rank"] + model_results["r4"]
        else: 
            model_results["sum_rank"] =  model_results["r4"]
        
        if not ("max_rank" in model_results.columns):
            model_results["max_rank"] =  model_results["r4"]
        else: 
            model_results['max_rank'] = model_results[['max_rank', "r4"]].max(axis=1)

        
        if 'level_0' in model_results.columns:
            model_results = model_results.drop(columns="level_0")

    return model_results.sort_values("sum_rank").reset_index().drop(columns="level_0")


def load_abund_model(name, k, size=(1,40,20)):
    """
    Loads a SimpleNN-abund model.
    """
    model = ADModel_abund(size, k)
    model.load_state_dict(torch.load(f"{name}.pth"))
    return model

def load_act_model(file, kernel_size, size=(1,40,20)):
    """
    Loads a SimpleNN-GFP model.
    """
    model = ADModel_act(size,kernel_size)
    model.load_state_dict(torch.load(f"{file}.pth"))

    model.eval()
    return model

def load_two_state_model(row, size=(1, 40, 20)):
    """
    Loads a three state BiophysicalNN model.
    """
    model_name = row.loc[0, "model_name"].removeprefix("../../")
    kernel_size_three = row.loc[0, "kernel_size"]
    abund_k_three = int(re.search(r'ak(\d+)',model_name).group(1))
    activity_fun = row.loc[0, "activity_fun"]

    model = ADModel_three_state_abund(size,activity_fun, kernel_size_three, relu=False, abund_k=abund_k_three)
    model.load_state_dict(torch.load(f"{model_name}.pth"))

    model.eval()
    return model, abund_k_three

def get_conv_weights(model, layer): 
    """
    Extracts the convolutional layer weights and
    returns them as a pandas dataframe
    """
    model_state_dict = model.state_dict()
    width = model_state_dict[layer].shape[0] * model_state_dict[layer].shape[2]
    conv_weight = model_state_dict[layer].detach().numpy().reshape(width,20)
    conv_weights = pd.DataFrame(conv_weight,columns = ["R","H","K","D","E","S","T","N","Q","C","G","P","A","V","I","L","M","F","Y","W"])

    # Order of AAs for plotting
    amino_acids = ["R","K","H","D","E","M","I","L","V","A","F","Y","W","Q","N","S","T","G","P","C"]

    conv_weights = conv_weights[amino_acids]
    return conv_weights

def get_linear_weights(model, layer, kernel_size, max_width=40):
    """
    Extracts the linear layer weights and
    returns them as a numpy array. Also returns the 
    corresponding x-axis values
    """
    linear_weights = (max_width - kernel_size + 1)
    model_state_dict = model.state_dict()
    linear_weight = model_state_dict[layer].detach().numpy().reshape(linear_weights)
    x_vals = np.array(range(0,linear_weights))
    return linear_weight, x_vals

def make_row_plot(fig, kernel_size, model, row, lim, height_ratios, num_rows):
    """
    Core functionality for creating the convolutional filter plots. 
    Plots the convolution filter as a heatmap and the linear weights as a bar plot.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure object created with plt.figure
    
    kernel_size : int
        The size of the convolutional filter

    model : ADModel_abund
        Loaded model (from which parameters will be extracted)
    
    row : int
        Which row in the overall figure should we fill in

    lim : float
        The max value for the heatmap

    height_ratios : list
        Ratio of heights between rows, length should correspond to number of rows in
        gridspec object

    num_rows : list
        The number of rows in the gridspec object

    Returns
    -------
    ax1 : maplotlib.Axes object
        Contains the heatmap
    ax2 : maplotlib.Axes object
        Contains the barplot
    im1 : maplotlib.Axes.imshow object
        The heatmap
    """
    # Setting the color arguments
    # This is setting up our colors for the plots

    # Colors for convolutional filter
    colors = [(1, 0, 0), (1, 1, 1), (0, 0, 1)]  # Blue, White, Red
    n_bins = 100  # Number of color bins
    cmap_name = "custom_colormap"
    custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
    custom_cmap.set_bad(color='white', alpha=1.0)  # Handle NaN values, if any

    # Colors for AA labels
    colors = {'A': 'purple', 'C': 'darkorange', 'E': 'blue', 'D': 'blue', 'G': 'darkorange', 
                  'F': 'indigo', 'I': 'purple', 'H': 'green', 'K': 'green', 'M': 'purple', 
                  'L': 'purple', 'N': 'darkslategray', 'Q': 'darkslategray', 'P': 'darkorange', 'S': 'darkslategray', 
                  'R': 'green', 'T': 'darkslategray', 'W': 'indigo', 'V': 'purple', 'Y': 'indigo'}

    # Load the convolutional and linear weights
    model_conv1 = get_conv_weights(model, "conv1.weight").T
    model_linear1, model_xvals1 = get_linear_weights(model, "linear1.weight", kernel_size)

    # Calculate how much of the figure the convolutional filter vs. linear weights will take up
    ratio1 = kernel_size 
    ratio2 = 40 - kernel_size + 1

    # GridSpec allows us to plot rows/columns with different dimensions
    # Here, we initalize the gridspec object with the total number of rows in our figure
    # But we only fill in the specific row that we are interested in
    # (When combined with many other gridspec objects, they will all line up and look good)
    gs1 = gridspec.GridSpec(2 + num_rows*2, 2, width_ratios=[ratio1, ratio2], height_ratios=height_ratios, figure=fig)

    # ax1 and ax2 are matplotlib axes
    ax1 = plt.subplot(gs1[row, 0])
    ax2 = plt.subplot(gs1[row, 1])

    # This is the code for plotting the convolutional filter as a heatmap
    # Min and max values of heatmap are passed by used
    im1 = ax1.imshow(model_conv1.to_numpy(), cmap=custom_cmap, aspect='auto',vmin=-lim,vmax=lim)
    ax1.set_yticks(range(len(model_conv1.index)), model_conv1.index)  # Label x-axis with column names
    ax1.set_xticks(np.arange(kernel_size, step=2))
    
    # This is to color the amino acid labels
    for ytic in ax1.get_yticklabels():
        if ytic.get_text() in colors.keys(): 
            ytic.set_color(colors[ytic.get_text()])
    ax1.tick_params(axis='x', which='major', labelsize=15) 
    ax1.tick_params(axis='y', which='major', labelsize=8) 

    # This is the code for plotting the linear weights as a bar graph
    ax2.bar(model_xvals1, model_linear1, color="grey")
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["bottom"].set_position('zero')
    ax2.spines["bottom"].set_linewidth(2.5)

    ax2.spines['bottom'].set_bounds(min(model_xvals1) - 1, max(model_xvals1) + 1)
    
    # Remove all x-ticks for the bargraph
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])

    # Return axes objects
    return ax1, ax2, im1

def make_row_plot_abund(fig, kernel_size, model, row, lim, height_ratios, num_rows, abund_kernel_size, max_width=40):
    """
    Core functionality for creating the convolutional filter plots. 
    Plots the convolution filter as a heatmap and the linear weights as a bar plot.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure object created with plt.figure
    
    kernel_size : int
        The size of the convolutional filter

    model : ADModel_abund
        Loaded model (from which parameters will be extracted)
    
    row : int
        Which row in the overall figure should we fill in

    lim : float
        The max value for the heatmap

    height_ratios : list
        Ratio of heights between rows, length should correspond to number of rows in
        gridspec object

    num_rows : list
        The number of rows in the gridspec object

    Returns
    -------
    ax1 : maplotlib.Axes object
        Contains the heatmap
    ax2 : maplotlib.Axes object
        Contains the barplot
    im1 : maplotlib.Axes.imshow object
        The heatmap
    """
    colors = [(1, 0, 0), (1, 1, 1), (0, 0, 1)]  # Blue, White, Red
    n_bins = 100  # Number of color bins
    cmap_name = "custom_colormap"
    custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
    custom_cmap.set_bad(color='white', alpha=1.0)  # Handle NaN values, if any

    # Colors for AA labels
    colors = {'A': 'purple', 'C': 'darkorange', 'E': 'blue', 'D': 'blue', 'G': 'darkorange', 
                  'F': 'indigo', 'I': 'purple', 'H': 'green', 'K': 'green', 'M': 'purple', 
                  'L': 'purple', 'N': 'darkslategray', 'Q': 'darkslategray', 'P': 'darkorange', 'S': 'darkslategray', 
                  'R': 'green', 'T': 'darkslategray', 'W': 'indigo', 'V': 'purple', 'Y': 'indigo'}
    
    # Load the convolutional and linear weights for both NN blocks
    model_conv1 = get_conv_weights(model, "conv1.weight").T
    model_linear1, model_xvals1 = get_linear_weights(model, "linear1.weight", kernel_size, max_width=max_width)

    model_conv2 = get_conv_weights(model, "conv2.weight").T
    model_linear2, model_xvals2 = get_linear_weights(model, "linear2.weight", abund_kernel_size, max_width=max_width)

    # Calculate how much of the figure the convolutional filters vs. linear weights will take up
    ratio1 = kernel_size 
    ratio2 = max_width - kernel_size + 1
    ratio1_abund = abund_kernel_size
    ratio2_abund = max_width - abund_kernel_size + 1

    # GridSpec allows us to plot rows/columns with different dimensions
    # Here, we initalize the gridspec object with the total number of rows in our figure
    # But we only fill in the specific row that we are interested in
    # (When combined with many other gridspec objects, they will all line up and look good)
    gs1 = gridspec.GridSpec(2 + num_rows, 5, width_ratios=[ratio1, ratio2, 0.5, ratio1_abund, ratio2_abund], height_ratios=height_ratios, figure=fig)
    
    # axs are matplotlib axes
    ax1 = plt.subplot(gs1[row, 0])
    ax2 = plt.subplot(gs1[row, 1])
    ax3 = plt.subplot(gs1[row, 3])
    ax4 = plt.subplot(gs1[row, 4])


    ## Code for first set of parameters --> K1 predictor
    # This is the code for plotting the convolutional filter as a heatmap
    # Min and max values of heatmap are passed by used
    im1 = ax1.imshow(model_conv1.to_numpy(), cmap=custom_cmap, aspect='auto',vmin=-lim,vmax=lim)
    ax1.set_yticks(range(len(model_conv1.index)), model_conv1.index)  # Label x-axis with column names
    ax1.set_xticks(np.arange(kernel_size, step=2))

    # This is to color the amino acid labels
    for ytic in ax1.get_yticklabels():
        if ytic.get_text() in colors.keys(): 
            ytic.set_color(colors[ytic.get_text()])
    ax1.tick_params(axis='both', which='major', labelsize=6) 

    # This is the code for plotting the linear weights as a bar graph
    ax2.bar(model_xvals1, model_linear1, color="grey")
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["bottom"].set_position('zero')
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])

    ## Code for second set of parameters --> abundance predictor
    # This is the code for plotting the convolutional filter as a heatmap
    # Min and max values of heatmap are passed by used
    im2 = ax3.imshow(model_conv2.to_numpy(), cmap=custom_cmap, aspect='auto',vmin=-lim,vmax=lim)
    ax3.set_yticks(range(len(model_conv2.index)), model_conv2.index)  # Label x-axis with column names
    ax3.set_xticks(np.arange(abund_kernel_size, step=2))
    
    # This is to color the amino acid labels
    for ytic in ax3.get_yticklabels():
        if ytic.get_text() in colors.keys(): 
            ytic.set_color(colors[ytic.get_text()])
    ax3.tick_params(axis='both', which='major', labelsize=6) 

    # This is the code for plotting the linear weights as a bar graph
    print(model_xvals2, model_linear2)
    ax4.bar(model_xvals2, model_linear2, color="grey")
    ax4.spines["top"].set_visible(False)
    ax4.spines["left"].set_visible(False)
    ax4.spines["right"].set_visible(False)
    ax4.spines["bottom"].set_position('zero')
    ax4.set_xticks([])
    ax4.set_yticks([])
    ax4.set_xticklabels([])
    ax4.set_yticklabels([])


    return ax1, ax2, ax3, ax4, im1, im2
