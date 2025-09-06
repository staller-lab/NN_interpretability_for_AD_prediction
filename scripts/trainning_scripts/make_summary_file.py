# Write results to a file
import torch
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import argparse
import os
from scipy.stats import pearsonr, spearmanr

import sys
sys.path.append("/Users/claireleblanc/Documents/grad_school/staller_lab/NN_interpretability_for_AD_prediction/Model")
from ADModel_three_state import ADModel_three_state, ADModel_three_state_abund
from ADModel_two_state import ADModel_two_state, ADModel_two_state_abund
from ADModel_act import ADModel_act
from ADModel_abund import ADModel_abund
from Data import DataReader, SplitData, FastTensorDataLoader

torch.manual_seed(25)

# For higher resoltion figures
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

parser = argparse.ArgumentParser()

# Adding arguments - these correspond to those in Model.py
parser.add_argument("-f","--file",help="File with activity, sequence, abundance data",type=str, default=None)
parser.add_argument("-s", "--scale",help = "Value to divide activity and abundance by",type=str, default=None)
parser.add_argument("-a", "--activity", help = "Function used to calcualte activity", type=str, default=None)
parser.add_argument("-c", "--outchannels", help = "File to write predicted data", type=int, default=None)
parser.add_argument("-k","--kernel_size",help="Width of kernel",type=int, default=None)
parser.add_argument("-r","--relu",action='store_true',help="Use the ReLU activation function instead of the parametric relu")
parser.add_argument("-m","--model",help="Path to trained model",type=str, default=None)
parser.add_argument("-i","--intelligent_split",action='store_true',help="Whether to use separate file of validation data (must be provided with the -v argument)")
parser.add_argument("-v", "--val_file", help = "Input file to validation sequences", type=str, default=None)
parser.add_argument("-t", "--test_file", help = "Input file to test sequences", type=str, default=None)
parser.add_argument("-n","--normal_model",help="Which model to use. Options are: closed, abundance, or two-state", type=str, default=None)
parser.add_argument("-np",help="penalty for negative k/closed values", type=float, default=None)
parser.add_argument("-wp",help="penalty for negative weight values", type=float, default=None)
parser.add_argument("-hv", "--hill_value", help="What n value to use in the hill function", type=int, default=None)
parser.add_argument("-ak", "--abund_kernel_value", help="What kernel size to use to predict the activity", type=int, default=None)
parser.add_argument("-rf", "--results_file", help="Where to store the results", type=str, default=None)

args = parser.parse_args()

file = args.file

# Saving arguments 
if args.val_file: 
    val_file = args.val_file

if args.test_file: 
    test_file = args.test_file

if args.scale:
    scale = args.scale
else:
    scale = 1000
batch_size = -1 # One big batch
if args.activity is not None:
    activity_fun = args.activity
else:
    activity_fun = "Linear"
if args.kernel_size: 
    kernel_size = args.kernel_size
else: 
    kernel_size = 10
if args.outchannels:
    outchannel = args.outchannels
else:
    outchannel = 1
if args.hill_value:
    hill_val = args.hill_value
else:
    hill_val=1
if args.abund_kernel_value:
    abund_kernel_value = args.abund_kernel_value
else:
    abund_kernel_value = 15

model = args.model
smart_split = args.intelligent_split
relu = args.relu

data_reader = DataReader()

# Loading the data

# Creating the split data object
split_data = SplitData(data_reader,encoding_type="2D")

if smart_split: 
    X, y_abundance, y_activity, size = split_data.read_split_data(file, val_file, test_file, scaler=scale)

else: 
    X, y_abundance, y_activity, size = split_data.read_data(file, scale=scale)

tensor_X = torch.from_numpy(X).type(torch.FloatTensor)
tensor_y_abund = torch.from_numpy(np.array(y_abundance)).type(torch.FloatTensor)
tensor_y_activity = torch.from_numpy(np.array(y_activity)).type(torch.FloatTensor)
tensor_y = torch.stack((tensor_y_abund, tensor_y_activity)).transpose(0,1)

# data = FastTensorDataLoader(tensor_X, tensor_y, batch_size=len(tensor_X), shuffle=False)

# Default arguments
size = tuple(tensor_X.shape[1:4]) # Get second two dimensions (first dimension is # of samples)
two_state = False
abund_kernel = False
simple_act = False
simple_abund = False

# Loading the model 
if args.normal_model == "three_state": 
    loaded_model = ADModel_three_state(size,activity_fun,kernel_size,outchannel,relu,hill_val)  # Create an instance of your model
elif args.normal_model == "three_state_abund": 
    abund_kernel = True
    loaded_model = ADModel_three_state_abund(size,activity_fun,kernel_size,outchannel,relu,hill_val, abund_kernel_value)
elif args.normal_model == "two_state": 
    two_state = True
    loaded_model = ADModel_two_state(size,activity_fun,kernel_size,outchannel,relu,hill_val)
elif args.normal_model == "two_state_abund": 
    abund_kernel = True
    two_state = True
    loaded_model = ADModel_two_state_abund(size,activity_fun,kernel_size,outchannel,relu,hill_val, abund_kernel_value)
elif args.normal_model == "simple_act":
    simple_act = True
    loaded_model = ADModel_act(size,kernel_size)
elif args.normal_model == "simple_abund":
    simple_abund = True
    loaded_model = ADModel_abund(size,kernel_size)
else:
    print("Defaulting to three state model")
    loaded_model = ADModel_three_state(size,activity_fun,kernel_size,outchannel,relu,hill_val)  # Create an instance of your model

loaded_model.load_state_dict(torch.load(f"{model}.pth"))

loaded_model.eval()
pred_vals = loaded_model(tensor_X) # Need to predict values in order to get Ks

# Things to write
# file_name, kernel_size, activity_fun, activity_weighting, hill value, r^2 values, pearson values, K1 negative?, K2 negative? 

# Read in the predictions
vals_all = pd.read_csv(f"{model}_vals.csv")
train_all = pd.read_csv(f"{model}_train.csv")
test_all = pd.read_csv(f"{model}_test.csv")

# Count the total number of parameters
total_params = 0
for name, parameter in loaded_model.named_parameters():
    if not parameter.requires_grad:
        continue
    params = parameter.numel()
    total_params += params

if simple_act: 
    spearman_act_test = spearmanr(test_all["activity_actual"],test_all["activity_pred"]).correlation
    spearman_act_val = spearmanr(vals_all["activity_actual"],vals_all["activity_pred"]).correlation
    spearman_act_train = spearmanr(train_all["activity_actual"],train_all["activity_pred"]).correlation

    pearson_act_test = pearsonr(test_all["activity_actual"],test_all["activity_pred"]).correlation
    pearson_act_val = pearsonr(vals_all["activity_actual"],vals_all["activity_pred"]).correlation
    pearson_act_train = pearsonr(train_all["activity_actual"],train_all["activity_pred"]).correlation

    # The columns are:
    # The model name, the model type, the kernel size, the activity function, whether to penalize negative Ks
    # Whether to penalize negative weights, hill value, spearman abund test, spearman abund val, spearman abund train,
    # spearman act test, spearman act val, spearman act train, pearson abund test, pearson abund val, pearson abund train, 
    # pearson act test, pearson act val, pearson act train, whether K1 is negative, whether K2 is negative, whether abund
    # is negative, Pearson or spearman relu activation, total params
    string_to_write = f"{model},{args.normal_model},{kernel_size},NA,NA,{args.wp},NA,NA,{spearman_act_test},NA,{spearman_act_val},NA,{spearman_act_train},NA,{pearson_act_test},NA,{pearson_act_val},NA,{pearson_act_train},NA,NA,NA,NA,{total_params}\n"

elif simple_abund: 
    spearman_abund_test = spearmanr(test_all["abund_actual"],test_all["abund_pred"]).correlation
    spearman_abund_val = spearmanr(vals_all["abund_actual"],vals_all["abund_pred"]).correlation
    spearman_abund_train = spearmanr(train_all["abund_actual"],train_all["abund_pred"]).correlation
    pearson_abund_test = pearsonr(test_all["abund_actual"],test_all["abund_pred"]).correlation
    pearson_abund_val = pearsonr(vals_all["abund_actual"],vals_all["abund_pred"]).correlation
    pearson_abund_train = pearsonr(train_all["abund_actual"],train_all["abund_pred"]).correlation

    string_to_write = f"{model},{args.normal_model},{kernel_size},NA,NA,{args.wp},NA,{spearman_abund_test},NA,{spearman_abund_val},NA,{spearman_abund_train},NA,{pearson_abund_test},NA,{pearson_abund_val},NA,{pearson_abund_train},NA,NA,NA,NA,NA,{total_params}\n"

else:
    if not two_state: 
        k1s, k2s, closed = loaded_model.get_ks()
        k1_negative = np.any(k1s.detach().numpy() < 0)
        k2_negative = np.any(k2s.detach().numpy() < 0)
        closed_negative = np.any(closed.detach().numpy() < 0)
    else:
        k1s, inactive = loaded_model.get_ks()
        k1_negative = np.any(k1s.detach().numpy() < 0)
        k2_negative = "NA"
        closed_negative = np.any(inactive.detach().numpy() < 0)

    spearman_act_test = spearmanr(test_all["activity_actual"],test_all["activity_pred"]).correlation
    spearman_act_val = spearmanr(vals_all["activity_actual"],vals_all["activity_pred"]).correlation
    spearman_act_train = spearmanr(train_all["activity_actual"],train_all["activity_pred"]).correlation
    pearson_act_test = pearsonr(test_all["activity_actual"],test_all["activity_pred"]).correlation
    pearson_act_val = pearsonr(vals_all["activity_actual"],vals_all["activity_pred"]).correlation
    pearson_act_train = pearsonr(train_all["activity_actual"],train_all["activity_pred"]).correlation

    spearman_abund_test = spearmanr(test_all["abund_actual"],test_all["abund_pred"]).correlation
    spearman_abund_val = spearmanr(vals_all["abund_actual"],vals_all["abund_pred"]).correlation
    spearman_abund_train = spearmanr(train_all["abund_actual"],train_all["abund_pred"]).correlation
    pearson_abund_test = pearsonr(test_all["abund_actual"],test_all["abund_pred"]).correlation
    pearson_abund_val = pearsonr(vals_all["abund_actual"],vals_all["abund_pred"]).correlation
    pearson_abund_train = pearsonr(train_all["abund_actual"],train_all["abund_pred"]).correlation

    string_to_write = f"{model},{args.normal_model},{kernel_size},{activity_fun},{args.np},{args.wp},{hill_val},{spearman_abund_test},{spearman_act_test},{spearman_abund_val},{spearman_act_val},{spearman_abund_train},{spearman_act_train},{pearson_abund_test},{pearson_act_test},{pearson_abund_val},{pearson_act_val},{pearson_abund_train},{pearson_act_train},{k1_negative},{k2_negative},{closed_negative},{relu},{total_params}\n"

filename = args.results_file

if not os.path.exists(filename):
    with open(filename, 'a') as f: 
        f.write("model_name,model_type,kernel_size,activity_fun,negative_pen,weight_pen,hill_value,spearman_abund_test,spearman_act_test,spearman_abund_val,spearman_act_val,spearman_abund_train,spearman_act_train,pearson_abund_test,pearson_act_test,pearson_abund_val,pearson_act_val,pearson_abund_train,pearson_act_train,K1_negative,K2_negative,abund_negative,relu,total_params\n")


with open(filename, 'a+') as f:
    f.write(string_to_write)

