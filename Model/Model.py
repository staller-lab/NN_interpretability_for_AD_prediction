"""
Modified Date: 29 August 2025
Author: Claire LeBlanc

Usage: python Final_Model.py [optional args]
"""

import torch
from Data import DataReader, SplitData
from ADModel_act import ADModel_act
from ADModel_abund import ADModel_abund
from ADModel_three_state import ADModel_three_state, ADModel_three_state_abund
from ADModel_two_state import ADModel_two_state, ADModel_two_state_abund

import pandas as pd
import matplotlib.pyplot as plt
import argparse
import torch.nn.init as init
import copy
import numpy as np
from scipy.stats import pearsonr
torch.manual_seed(25)


def get_params(args):
    """
    Reads in command line arguments and stores them as variables.
    
    Parameters 
    ----------
    args : argparse object
        Command line argument object

    Returns
    -------
    file : str
        Path to input data file (or input trainning file if intelligent split)

    val_file : str
        Path to validation data file
    
    intelligent_split : str
        True indicates user will provide train/validation split. Otherwise split is random

    scale : str
        Method for scaling the activity and abundance
    
    batch_size : int
        Size of trainning batches
    
    activity_fun : str
        Function used to go from bound to activity

    epochs : int
        Number of epochs to train for
    
    outfile : str
        File prefix to save all related files
    
    outchannel : int
        Number of channels in convolutional filter
    
    kernel_size : int
        Size on the convolutional filters

    abund_kernel_value : int
        Size of the abundance convolutional filters, only used in some NNs

    learning_rate : float
        The learning rate for trainning

    relu : bool
        True indicates using the ReLU activation function, otherwise parametric ReLU is used

    test : bool
        True indicates also evaluating on test data

    neg_pen : float
        How much to penalize negative intermediate biophysical values

    weight_pen : float
        How much to penalize negative linear weights

    hill_val : int
        Hill coefficient
    
    seed : int
        The random seed used to initalize NN weights
    """ 
    if args.file:
        file = args.file
    else:
        file = "/Users/claireleblanc/Documents/grad_school/rotation 1/code/Model_TF_States/Data/pm_gcn4_sort2_pools_allchannels_wrangled.csv"

    if args.val_file:
        val_file = args.val_file
    if args.scale:
        scale = args.scale
    else:
        scale = 1000
    if args.outfile:
        outfile = args.outfile
    else:
        outfile = "pred_actual.csv"
    if args.batch:
        batch_size = args.batch
    else: 
        batch_size = 100
    if args.learning_rate:
        learning_rate = args.learning_rate
    else:
        learning_rate = 0.0001
    if args.activity:
        activity_fun = args.activity
    else: 
        activity_fun = "Linear"
    if args.epochs:
        epochs = args.epochs
    else:
        epochs = 2
    if args.outchannels:
        outchannel = args.outchannels
    else:
        outchannel = 1
    if args.kernel_size:
        kernel_size = args.kernel_size
    else:
        kernel_size = 10

    intelligent_split = args.intelligent_split
    relu = args.relu
    test = args.test
    
    if args.negative_penalty:
        neg_pen = args.negative_penalty
    else:
        neg_pen = 0 

    if args.weight_penalty:
        weight_pen = args.weight_penalty
    else:
        weight_pen = 0 

    if args.hill_value:
        hill_val = args.hill_value
    else:
        hill_val = 2
    if args.abund_kernel_value:
        abund_kernel_value = args.abund_kernel_value
    else:
        abund_kernel_value = 15

    if args.seed:
        seed = args.seed
    else:
        seed = 25

    return file, scale, batch_size, activity_fun, epochs, outfile, outchannel, kernel_size, learning_rate, relu, test, val_file, intelligent_split, neg_pen, weight_pen, hill_val, abund_kernel_value, seed

# Run the model in the normal way
def fit(args): 
    """
    Loads the data, trains the model, gets the actual and predicted values, saves the model, and plots the losses.
    
    Parameters
    ----------
    args : tuple
        Command line arguments read in by get_params

    """
    
    params = get_params(args)
    file = params[0]
    scale = params[1] 
    batch_size = params[2]
    activity_fun = params[3]
    epochs = params[4]
    outfile = params[5]
    outchannel = params[6]
    kernel_size = params[7]
    learning_rate = params[8]
    relu = params[9] 
    test_file = params[10] 
    val_file = params[11]
    intelligent_split = params[12]
    neg_pen = params[13]
    weight_pen = params[14] 
    hill_val = params[15]
    abund_kernel_value = params[16]
    seed = params[17]

    # Loading in the data 
    data_reader = DataReader()
    
    # Split data into train and testing sets and create data loaders for each group
    split_data = SplitData(data_reader,encoding_type="2D")
    
    if intelligent_split: 
        split_data.read_split_data(file, val_file, test_file, scaler=scale)
    else: 
        split_data.read_data(scale=scale)
        split_data.split_data(ratio=False)

    train_data_loader, val_data_loader, test_data_loader, size = split_data.load_data(batch_size)
    
    # Initalizing the model
    if args.model == "three_state":
        print(f"Training three state model with filter size of {kernel_size}")
        model = ADModel_three_state(size, activity_fun, kernel_size, outchannel, relu, hill_val, seed=seed)
    elif args.model == "three_state_abund":
        print(f"Training three state model with filter size of {kernel_size} and abundance filter size of {abund_kernel_value}")
        model = ADModel_three_state_abund(size, activity_fun, kernel_size, outchannel, relu, hill_val, abund_k=abund_kernel_value, seed=seed)
    elif args.model == "two_state":
        print(f"Training two state model with filter size of {kernel_size}")
        model = ADModel_two_state(size, activity_fun, kernel_size, outchannel, relu, hill_val, seed=seed)
    elif args.model == "two_state_abund":
        print(f"Training two state model with filter size of {kernel_size} and abundance filter size of {abund_kernel_value}")
        model = ADModel_two_state_abund(size, activity_fun, kernel_size, outchannel, relu, hill_val, abund_k=abund_kernel_value, seed=seed)
    elif args.model == "simple_act":
        print(f"Training simple activity model with filter size of {kernel_size}")
        model = ADModel_act(size,kernel_size, seed=seed)
    elif args.model == "simple_abund":
        print(f"Training simple abundance model with filter size of {kernel_size}")
        model = ADModel_abund(size,kernel_size, seed=seed)
    else: 
        print(f"Defaulting to three_state model with filter size of {kernel_size}")
        model = ADModel_three_state(size,activity_fun,kernel_size,outchannel,relu,hill_val, seed=seed)

    # Use GPUs if available --> I'm not sure this works
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)
    
    model.to(device)

    # Training the model
    if (args.model == "simple_act") or (args.model == "simple_abund"):
        print("Training simple module")
        model.train_model(train_data_loader,val_data_loader,epochs,device,learning_rate,outfile, weight_pen)
    else:
        # Trains the abudnance and activity predictors separately 
        print("Training abundance module")
        model.freeze_act_weights()
        model.train_model_abund(train_data_loader,val_data_loader,epochs,device,learning_rate,outfile, 0, 1)

        model.eval()

        print("Training activity module")
        model.train()
        model.freeze_abund_weights()
        model.unfreeze_act_weights()
        model.train_model_act(train_data_loader,val_data_loader,epochs,device,learning_rate,outfile, neg_pen, weight_pen)
        model.freeze_act_weights()

    model.eval()
    # Get final model predictions
    model.get_actual_and_predicted(val_data_loader, device,filepath=f"{outfile}_vals.csv")
    model.get_actual_and_predicted(train_data_loader, device, filepath=f"{outfile}_train.csv")

    # Need to implement testing!!!
    if test_file:
        model.get_actual_and_predicted(test_data_loader, device,filepath=f"{outfile}_test.csv")

    # Plot test and train losses
    model.get_losses(outfile)

def main():
    """
    Reads in command line arguments and calls the main workhorse function (fit)
    """
    # Initialize parser
    parser = argparse.ArgumentParser()

    # Adding arguments related to model training
    parser.add_argument("-f", "--file", help = "Input file", type=str)
    parser.add_argument("-v", "--val_file", help = "Input file to validation sequences", type=str)
    parser.add_argument("-b", "--batch", help = "Batch size",type=int)
    parser.add_argument("-l", "--learning_rate", help = "Learning rate",type=float)
    parser.add_argument("-e", "--epochs", help = "Number of epochs", type=int)
    parser.add_argument("-o", "--outfile", help = "File to write predicted data", type=str)
    parser.add_argument("-t","--test", help="Test file to evaluate the data on", type=str)
    parser.add_argument("-i","--intelligent_split",action='store_true',help="Whether to use separate file of validation data (must be provided with the -v argument)")
    parser.add_argument("-np","--negative_penalty",help="How much to penalize negative K values",type=float)
    parser.add_argument("-wp","--weight_penalty",help="How much to penalize negative weight values", type=float)

    # Adding arguments related to the actual model
    parser.add_argument("-s", "--scale",help = "Method for scaling the activity and abundance",type=str)
    parser.add_argument("-a", "--activity", help = "Function used to calcualte activity", type=str)
    parser.add_argument("-c", "--outchannels", help = "File to write predicted data", type=int)
    parser.add_argument("-k","--kernel_size",help="Width of kernel",type=int)
    parser.add_argument("-r","--relu",action='store_true',help="Use the ReLU activation function instead of the parametric relu")
    parser.add_argument("-m","--model",help="Which model to run. Options are simple_abund, simple_act, three_state_abund, two_state_abund, three_state, two_state",type=str)
    parser.add_argument("-hv", "--hill_value", help="What n value to use in the hill function", type=int)
    parser.add_argument("-ak", "--abund_kernel_value", help="What kernel size to use to predict the activity", type=int)

    # seed
    parser.add_argument("-seed", "--seed", help="Initalization seed to use", type=int)


    # Read arguments from command line
    args = parser.parse_args()
    fit(args)
    
if __name__ == "__main__":
    main()