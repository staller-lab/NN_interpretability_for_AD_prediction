# Write results to a file
import torch
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import argparse
import os
from scipy.stats import pearsonr

from Model import ADModel_three_state, ADModel_three_state_abund, ADModel_two_state, ADModel_two_state_abund, ADModel_act, ADModel_abund
from Data import DataReader, SplitData, FastTensorDataLoader


torch.manual_seed(25)

# For higher resoltion figures
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

parser = argparse.ArgumentParser()

parser.add_argument("-f","--file",help="File with activity, sequence, abundance data",type=str, default=None)
parser.add_argument("-s", "--scale",help = "Value to divide activity and abundance by",type=str, default=None)
parser.add_argument("-a", "--activity", help = "Function used to calcualte activity", type=str, default=None)
parser.add_argument("-aw", "--activity_weight", help = "Value activity was weighted by", type=str, default=None)
parser.add_argument("-c", "--outchannels", help = "File to write predicted data", type=int, default=None)
parser.add_argument("-k","--kernel_size",help="Width of kernel",type=int, default=None)
parser.add_argument("-r","--relu",action='store_true',help="Use the ReLU activation function instead of the parametric relu")
parser.add_argument("-m","--model",help="Path to trained model",type=str, default=None)
parser.add_argument("-p","--positive",action='store_true',help="Use only positive linear weights")
parser.add_argument("-i","--intelligent_split",action='store_true',help="Whether to use separate file of validation data (must be provided with the -v argument)")
parser.add_argument("-v", "--val_file", help = "Input file to validation sequences", type=str, default=None)
parser.add_argument("-n","--normal_model",help="Which model to use. Options are: closed, abundance, or two-state", type=str, default=None)
parser.add_argument("-np",help="penalty for negative k/closed values", type=float, default=None)
parser.add_argument("-wp",help="penalty for negative weight values", type=float, default=None)
parser.add_argument("-hv", "--hill_value", help="What n value to use in the hill function", type=int, default=None)
parser.add_argument("-ak", "--abund_kernel_value", help="What kernel size to use to predict the activity", type=int, default=None)


args = parser.parse_args()

file = args.file
if args.val_file: 
    val_file = args.val_file


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
positive = args.positive
activity_weight = args.activity_weight
relu = args.relu

data_reader = DataReader()

# Loading the data
split_data = SplitData(data_reader,encoding_type="2D")

if smart_split: 
    X, y_abundance, y_activity, size = split_data.read_split_data(file, val_file, scaler=scale)

else: 
    X, y_abundance, y_activity, size = split_data.read_data(file, scale=scale)

tensor_X = torch.from_numpy(X).type(torch.FloatTensor)
tensor_y_abund = torch.from_numpy(np.array(y_abundance)).type(torch.FloatTensor)
tensor_y_activity = torch.from_numpy(np.array(y_activity)).type(torch.FloatTensor)
tensor_y = torch.stack((tensor_y_abund, tensor_y_activity)).transpose(0,1)

data = FastTensorDataLoader(tensor_X, tensor_y, batch_size=len(tensor_X), shuffle=False)

size = tuple(tensor_X.shape[1:4]) # Get second two dimensions (first dimension is # of samples)
two_state = False
abund_kernel = False
simple_act = False
simple_abund = False

if args.normal_model == "three_state": 
    loaded_model = ADModel_three_state(size,activity_fun,kernel_size,outchannel,relu,positive, hill_val)  # Create an instance of your model
elif args.normal_model == "three_state_abund": 
    abund_kernel = True
    loaded_model = ADModel_three_state_abund(size,activity_fun,kernel_size,outchannel,relu,positive, hill_val, abund_kernel_value)
elif args.normal_model == "two_state": 
    two_state = True
    loaded_model = ADModel_two_state(size,activity_fun,kernel_size,outchannel,relu,positive, hill_val)
elif args.normal_model == "two_state_abund": 
    abund_kernel = True
    two_state = True
    loaded_model = ADModel_two_state_abund(size,activity_fun,kernel_size,outchannel,relu,positive, hill_val, abund_kernel_value)
elif args.normal_model == "simple_act":
    simple_act = True
    loaded_model = ADModel_act(size,kernel_size)
elif args.normal_model == "simple_abund":
    simple_abund = True
    loaded_model = ADModel_abund(size,kernel_size)
else:
    print("Defaulting to three state model")
    loaded_model = ADModel_three_state(size,activity_fun,kernel_size,outchannel,relu,positive, hill_val)  # Create an instance of your model

loaded_model.load_state_dict(torch.load(f"{model}.pth"))

loaded_model.eval()
pred_vals = loaded_model(tensor_X) # Need to predict values in order to get Ks

# Things to write
# file_name, kernel_size, activity_fun, activity_weighting, hill value, r^2 values, pearson values, K1 negative?, K2 negative? 

vals_all = pd.read_csv(f"{model}_vals.csv")
train_all = pd.read_csv(f"{model}_train.csv")

total_params = 0
for name, parameter in loaded_model.named_parameters():
    if not parameter.requires_grad:
        continue
    params = parameter.numel()
    total_params += params

if simple_act: 
    Rsq_act_val = np.corrcoef(vals_all["activity_actual"],vals_all["activity_pred"])[0, 1]**2
    Rsq_act_train = np.corrcoef(train_all["activity_actual"],train_all["activity_pred"])[0, 1]**2
    pearson_act_val = pearsonr(vals_all["activity_actual"],vals_all["activity_pred"]).correlation
    pearson_act_train = pearsonr(train_all["activity_actual"],train_all["activity_pred"]).correlation

    string_to_write = f"{model},{args.normal_model},{kernel_size},NA,NA,{args.wp},NA,NA,NA,{Rsq_act_val},NA,{Rsq_act_train},NA,{pearson_act_val},NA,{pearson_act_train},NA,NA,NA,NA,{total_params}\n"

elif simple_abund: 
    Rsq_abund_val = np.corrcoef(vals_all["abund_actual"],vals_all["abund_pred"])[0, 1]**2
    Rsq_abund_train = np.corrcoef(train_all["abund_actual"],train_all["abund_pred"])[0, 1]**2
    pearson_abund_val = pearsonr(vals_all["abund_actual"],vals_all["abund_pred"]).correlation
    pearson_abund_train = pearsonr(train_all["abund_actual"],train_all["abund_pred"]).correlation

    string_to_write = f"{model},{args.normal_model},{kernel_size},NA,NA,{args.wp},NA,NA,{Rsq_abund_val},NA,{Rsq_abund_train},NA,{pearson_abund_val},NA,{pearson_abund_train},NA,NA,NA,NA,NA,{total_params}\n"

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

    Rsq_act_val = np.corrcoef(vals_all["activity_actual"],vals_all["activity_pred"])[0, 1]**2
    Rsq_act_train = np.corrcoef(train_all["activity_actual"],train_all["activity_pred"])[0, 1]**2
    pearson_act_val = pearsonr(vals_all["activity_actual"],vals_all["activity_pred"]).correlation
    pearson_act_train = pearsonr(train_all["activity_actual"],train_all["activity_pred"]).correlation

    Rsq_abund_val = np.corrcoef(vals_all["abund_actual"],vals_all["abund_pred"])[0, 1]**2
    Rsq_abund_train = np.corrcoef(train_all["abund_actual"],train_all["abund_pred"])[0, 1]**2
    pearson_abund_val = pearsonr(vals_all["abund_actual"],vals_all["abund_pred"]).correlation
    pearson_abund_train = pearsonr(train_all["abund_actual"],train_all["abund_pred"]).correlation

    string_to_write = f"{model},{args.normal_model},{kernel_size},{activity_fun},{args.np},{args.wp},{activity_weight},{hill_val},{Rsq_abund_val},{Rsq_act_val},{Rsq_abund_train},{Rsq_act_train},{pearson_abund_val},{pearson_act_val},{pearson_abund_train},{pearson_act_train},{k1_negative},{k2_negative},{closed_negative},{relu},{total_params}\n"

filename = "../results/results.csv"

if not os.path.exists(filename):
    with open(filename, 'a') as f: 
        f.write("model_name, model_type, kernel_size, activity_fun, negative_pen, weight_pen, activity_weight, hill_value, Rsq_abund_val, Rsq_act_val, Rsq_abund_train, Rsq_act_train, pearson_abund_val, pearson_act_val, pearson_abund_train, pearson_act_train, K1_negative, K2_negative, abund_negative, relu, total_params\n")


with open(filename, 'a+') as f:
    f.write(string_to_write)

