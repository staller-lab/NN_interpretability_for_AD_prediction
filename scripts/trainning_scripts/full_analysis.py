import torch
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import argparse
import os

os.chdir("/Users/claireleblanc/Documents/grad_school/staller_lab/NN_interpretability_for_AD_prediction/Model")
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

parser.add_argument("-f","--file",help="File with activity, sequence, abundance data",type=str, default=None)
parser.add_argument("-s", "--scale",help = "Value to divide activity and abundance by",type=str, default=None)
parser.add_argument("-a", "--activity", help = "Function used to calcualte activity", type=str, default=None)
parser.add_argument("-c", "--outchannels", help = "File to write predicted data", type=int, default=None)
parser.add_argument("-k","--kernel_size",help="Width of kernel",type=int, default=None)
parser.add_argument("-r","--relu",action='store_true',help="Use the ReLU activation function instead of the parametric relu")
parser.add_argument("-m","--model",help="Path to trained model",type=str, default=None)
parser.add_argument("-o","--output_dir",help="Output directory",type=str, default=None)
parser.add_argument("-p","--positive",action='store_true',help="Use only positive linear weights")
parser.add_argument("-i","--intelligent_split",action='store_true',help="Whether to use separate file of validation data (must be provided with the -v argument)")
parser.add_argument("-v", "--val_file", help = "Input file to validation sequences", type=str, default=None)
parser.add_argument("-t", "--test_file", help = "Input file to test sequences", type=str, default=None)
parser.add_argument("-n","--normal_model",help="Which model to use. Options are: closed, abundance, or two-state", type=str, default=None)
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
outdir = args.output_dir
smart_split = args.intelligent_split
relu = args.relu
positive = args.positive

data_reader = DataReader()

df = pd.read_csv(file)

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

abund_kernel = False
two_state = False
simple_act = False
simple_abund = False

if args.normal_model == "three_state": 
    loaded_model = ADModel_three_state(size,activity_fun,kernel_size,outchannel,relu,positive, hill_val)
elif args.normal_model == "three_state_abund":
    loaded_model = ADModel_three_state_abund(size,activity_fun,kernel_size,outchannel,relu,positive, hill_val, abund_kernel_value)
    abund_kernel = True
elif args.normal_model == "two_state": 
    two_state = True
    loaded_model = ADModel_two_state(size,activity_fun,kernel_size,outchannel,relu,positive, hill_val)
elif args.normal_model == "two_state_abund":
    two_state = True
    loaded_model = ADModel_two_state_abund(size,activity_fun,kernel_size,outchannel,relu,positive, hill_val, abund_kernel_value)
    abund_kernel = True
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

# Looking at predictions for all
pred_vals = loaded_model(tensor_X)

os.mkdir(outdir)

from prettytable import PrettyTable

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
    

pytorch_total_params = sum(p.numel() for p in loaded_model.parameters() if p.requires_grad)

with open(outdir + "/params.txt", "w+") as f:
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in loaded_model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    f.write(table.get_string())
    f.write(f"\nTotal Trainable Params: {total_params}")

# Making K histograms
if (not simple_act) and (not simple_abund): 
    if not two_state: 
        k1s, k2s, closed = loaded_model.get_ks()
        k1_df = pd.DataFrame(k1s.detach().numpy(),columns=["K1"])
        k2_df = pd.DataFrame(k2s.detach().numpy(),columns=["K2"])
        closed_df = pd.DataFrame(closed.detach().numpy(),columns=["closed"])
        predictions = pd.DataFrame(pred_vals.detach().numpy().reshape(-1, 2),columns=["predicted_activity","predicted_abundance"])
        new_df = pd.concat([df, k1_df,k2_df,closed_df,predictions],axis=1)
    else:
        K, inactive = loaded_model.get_ks()
        K_df = pd.DataFrame(K.detach().numpy(),columns=["K1"])
        inactive_df = pd.DataFrame(inactive.detach().numpy(),columns=["inactive"])
        predictions = pd.DataFrame(pred_vals.detach().numpy().reshape(-1, 2),columns=["predicted_activity","predicted_abundance"])
        new_df = pd.concat([df,K_df,inactive_df,predictions],axis=1)

    sorted_df = new_df.sort_values(by = "activity")
    lowest_activity = sorted_df[0:round(0.1*len(sorted_df))]
    top_activity = sorted_df[round(0.9*len(sorted_df)):]

    # Plotting K1
    plt.hist(top_activity["K1"], bins = 20, alpha=0.5, label="TFs with highest activity\n(top 10%)",density=True)
    plt.hist(lowest_activity["K1"], bins = 20, alpha=0.5, label = "TFs with lowest activity\n(bottom 10%)",density=True)
    plt.hist(sorted_df["K1"],bins=20, alpha=0.5, label = "All TFs",density=True)
    plt.legend(fontsize=10)
    plt.xlabel('K1 Value')
    plt.ylabel('Density')
    plt.title('Ratio of Closed to Open Transcription Factor')
    plt.savefig(f'{outdir}/K1.png')
    plt.clf()

    if not two_state:
        # Plotting K2
        plt.hist(top_activity["K2"], bins = 20, alpha=0.5, label="TFs with highest activity\n(top 10%)",density=True)
        plt.hist(lowest_activity["K2"], bins = 20, alpha=0.5, label = "TFs with lowest activity\n(bottom 10%)",density=True)
        plt.hist(sorted_df["K2"],bins=20, alpha=0.5, label = "All TFs",density=True)
        plt.legend(fontsize=10)
        plt.xlabel('K2 Value')
        plt.ylabel('Count')
        plt.title('Ratio of Open to Bound Transcription Factor')
        plt.savefig(f'{outdir}/K2.png')
        plt.clf()

        # Plotting closed
        plt.hist(top_activity["closed"], bins = 20, alpha=0.5, label="TFs with highest activity\n(top 10%)",density=True)
        plt.hist(lowest_activity["closed"], bins = 20, alpha=0.5, label = "TFs with lowest activity\n(bottom 10%)",density=True)
        plt.hist(sorted_df["closed"],bins=20, alpha=0.5, label = "All TFs",density=True)
        plt.legend(fontsize=10)
        plt.ylabel('Count')


        plt.xlabel('Abundance')
        plt.title('TF Abundance')
        plt.savefig(f'{outdir}/abundance.png')

        plt.clf()

    else:
        # Plotting closed
        plt.hist(top_activity["inactive"], bins = 20, alpha=0.5, label="TFs with highest activity\n(top 10%)",density=True)
        plt.hist(lowest_activity["inactive"], bins = 20, alpha=0.5, label = "TFs with lowest activity\n(bottom 10%)",density=True)
        plt.hist(sorted_df["inactive"],bins=20, alpha=0.5, label = "All TFs",density=True)
        plt.legend(fontsize=10)
        plt.xlabel('Inactive concentration')
        plt.ylabel('Count')
        plt.title('Amount of TF in the inactive state')
        plt.savefig(f'{outdir}/inactive.png')
        plt.clf()

# Weights
model_state_dict = loaded_model.state_dict()
width = model_state_dict['conv1.weight'].shape[0] * model_state_dict['conv1.weight'].shape[2]
colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]  # Blue, White, Red
n_bins = 100  # Number of color bins
cmap_name = "custom_colormap"
custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
custom_cmap.set_bad(color='white', alpha=1.0)  # Handle NaN values, if any
amino_acids = ["R","K","H","D","E","M","I","L","V","A","F","Y","W","Q","N","S","T","G","P","C"]
colors = {'A': 'purple', 'C': 'darkorange', 'E': 'blue', 'D': 'blue', 'G': 'darkorange', 
                  'F': 'indigo', 'I': 'purple', 'H': 'green', 'K': 'green', 'M': 'purple', 
                  'L': 'purple', 'N': 'darkslategray', 'Q': 'darkslategray', 'P': 'darkorange', 'S': 'darkslategray', 
                  'R': 'green', 'T': 'darkslategray', 'W': 'indigo', 'V': 'purple', 'Y': 'indigo'}

linear_weights = (40 - kernel_size + 1) * outchannel

# Conv1 weight analysis
conv1_weight = model_state_dict['conv1.weight'].detach().numpy().reshape(width,20)
conv1_weights = pd.DataFrame(conv1_weight,columns = ["R","H","K","D","E","S","T","N","Q","C","G","P","A","V","I","L","M","F","Y","W"]) # ["A","R","N","D","C","E","Q","G","H","I","L","K","M","F","P","S","T","W","Y","V"])
conv1_weights = conv1_weights[amino_acids]
fig, ax = plt.subplots()
lim = max(np.amax(conv1_weight),abs(np.amin(conv1_weight)))
im = ax.imshow(conv1_weights.to_numpy(), cmap=custom_cmap, aspect='auto',vmin=-lim,vmax=lim)
ax.set_xticks(range(len(conv1_weights.columns)), conv1_weights.columns)  # Label x-axis with column names
for xtic in ax.get_xticklabels():
    if xtic.get_text() in colors.keys(): 
        xtic.set_color(colors[xtic.get_text()])
fig.colorbar(im)
plt.title('Kernel for predicting K1')  # Set a title
plt.savefig(f'{outdir}/K1_kernel_weights.png')
plt.clf()

# Linear 1 weight analysis
linear1_weight = model_state_dict['linear1.weight'].detach().numpy().reshape(linear_weights)
# linear1_weight = model_state_dict['linear1.linear.weight'].detach().numpy().reshape(linear_weights)
x_vals = np.array(range(0,linear_weights))
plt.bar(x_vals,linear1_weight)
plt.xticks(fontsize=8)
plt.title("K1 Position Weighting")
plt.savefig(f'{outdir}/K1_seq_position_weights.png')
plt.clf()

if (not two_state) and (not simple_act) and (not simple_abund): 
    # Conv2 weight analysis
    width = model_state_dict['conv2.weight'].shape[0] * model_state_dict['conv2.weight'].shape[2]
    conv2_weight = model_state_dict['conv2.weight'].detach().numpy().reshape(width,20)
    conv2_weights = pd.DataFrame(conv2_weight,columns = ["R","H","K","D","E","S","T","N","Q","C","G","P","A","V","I","L","M","F","Y","W"]) # "A","R","N","D","C","E","Q","G","H","I","L","K","M","F","P","S","T","W","Y","V"])
    conv2_weights = conv2_weights[amino_acids]
    fig, ax = plt.subplots()
    lim = max(np.amax(conv2_weight),abs(np.amin(conv2_weight)))
    im = ax.imshow(conv2_weights.to_numpy(), cmap=custom_cmap, aspect='auto',vmin=-lim,vmax=lim)
    ax.set_xticks(range(len(conv2_weights.columns)), conv2_weights.columns)      
    for xtic in ax.get_xticklabels():
        if xtic.get_text() in colors.keys(): 
            xtic.set_color(colors[xtic.get_text()])
    fig.colorbar(im)
    plt.title('Kernel for predicting K2')  # Set a title
    plt.savefig(f'{outdir}/K2_kernel_weights.png')
    plt.clf()

    # Linear 2 weight analysis

    linear2_weight = model_state_dict['linear2.weight'].detach().numpy().reshape(linear_weights)
    x_vals = np.array(range(0,linear_weights))
    plt.bar(x_vals,linear2_weight)
    plt.xticks(fontsize=8)
    plt.title("K2 Position Weighting")
    plt.savefig(f'{outdir}/K2_seq_position_weights.png')
    plt.clf()

    # Conv3 weight analysis
    width = model_state_dict['conv3.weight'].shape[0] * model_state_dict['conv3.weight'].shape[2]
    conv3_weight = model_state_dict['conv3.weight'].detach().numpy().reshape(width,20)
    conv3_weights = pd.DataFrame(conv3_weight,columns =  ["R","H","K","D","E","S","T","N","Q","C","G","P","A","V","I","L","M","F","Y","W"]) # ["A","R","N","D","C","E","Q","G","H","I","L","K","M","F","P","S","T","W","Y","V"])
    conv3_weights = conv3_weights[amino_acids]
    fig, ax = plt.subplots()
    lim = max(np.amax(conv3_weight),abs(np.amin(conv3_weight)))
    im=ax.imshow(conv3_weights.to_numpy(), cmap=custom_cmap, aspect='auto',vmin=-lim,vmax=lim)
    ax.set_xticks(range(len(conv3_weights.columns)), conv3_weights.columns)  # Label x-axis with column names
    for xtic in ax.get_xticklabels():
        if xtic.get_text() in colors.keys(): 
            xtic.set_color(colors[xtic.get_text()])
    fig.colorbar(im)

    plt.title('Kernel for predicting abundance')  # Set a title
    plt.savefig(f'{outdir}/Abundance_kernel_weights.png')

    plt.clf()

    # Linear3 weight analysis

    if abund_kernel: 
        linear3_weight = model_state_dict['linear3.weight'].detach().numpy().reshape(40-args.abund_kernel_value+1)
        x_vals = np.array(range(0,40 -args.abund_kernel_value + 1))
    else:
        linear3_weight = model_state_dict['linear3.weight'].detach().numpy().reshape(linear_weights)
        x_vals = np.array(range(0,linear_weights))
    plt.bar(x_vals,linear3_weight)
    plt.xticks(fontsize=8)

    plt.title("Abundance Position Weighting")
    plt.savefig(f'{outdir}/abundance_seq_position_weights.png')

    plt.clf()
elif (not simple_act) and (not simple_abund): 
    # Conv2 weight analysis
    width = model_state_dict['conv2.weight'].shape[0] * model_state_dict['conv2.weight'].shape[2]
    conv2_weight = model_state_dict['conv2.weight'].detach().numpy().reshape(width,20)
    conv2_weights = pd.DataFrame(conv2_weight,columns = ["R","H","K","D","E","S","T","N","Q","C","G","P","A","V","I","L","M","F","Y","W"]) # "A","R","N","D","C","E","Q","G","H","I","L","K","M","F","P","S","T","W","Y","V"])
    conv2_weights = conv2_weights[amino_acids]
    fig, ax = plt.subplots()
    lim = max(np.amax(conv2_weight),abs(np.amin(conv2_weight)))
    im = ax.imshow(conv2_weights.to_numpy(), cmap=custom_cmap, aspect='auto',vmin=-lim,vmax=lim)
    ax.set_xticks(range(len(conv2_weights.columns)), conv2_weights.columns)      
    for xtic in ax.get_xticklabels():
        if xtic.get_text() in colors.keys(): 
            xtic.set_color(colors[xtic.get_text()])
    fig.colorbar(im)
    plt.title('Kernel for predicting inactive')  # Set a title
    plt.savefig(f'{outdir}/inactive_kernel_weights.png')
    plt.clf()

    # Linear 2 weight analysis

    if abund_kernel: 
        linear2_weight = model_state_dict['linear2.weight'].detach().numpy().reshape(40-args.abund_kernel_value+1)
        x_vals = np.array(range(0,40 -args.abund_kernel_value + 1))
    else:
        linear2_weight = model_state_dict['linear2.weight'].detach().numpy().reshape(linear_weights)
        x_vals = np.array(range(0,linear_weights))
    plt.bar(x_vals,linear2_weight)
    plt.xticks(fontsize=8)
    plt.title("inactive Position Weighting")
    plt.savefig(f'{outdir}/inactive_seq_position_weights.png')
    plt.clf()


# Scatter analysis - Validation
vals_all = pd.read_csv(f"{model}_vals.csv")
train_all = pd.read_csv(f"{model}_train.csv")

if (not simple_act) and (not simple_abund): 
    fig, ax = plt.subplots(1,2,figsize=[10,5])
    ax[0].scatter(vals_all["abund_actual"],vals_all["abund_pred"],color='C0', s=10, alpha=.3)
    Rsq = np.corrcoef(vals_all["abund_actual"],vals_all["abund_pred"])[0, 1]**2
    ax[0].set_xlabel('Measured Abundance')
    ax[0].set_ylabel('Predicted Abundance')
    ax[0].set_title(f'\n$R^2$={Rsq:.3}')
    xlim = [min(vals_all["abund_actual"]), max(vals_all["abund_actual"])]
    ax[0].plot(xlim, xlim, '--', color='k', label='diagonal', zorder=100)

    ax[1].scatter(vals_all["activity_actual"],vals_all["activity_pred"],color='C1', s=10, alpha=.3)
    Rsq = np.corrcoef(vals_all["activity_actual"],vals_all["activity_pred"])[0, 1]**2
    ax[1].set_xlabel('Measured Activity')
    ax[1].set_ylabel('Predicted Activity')
    ax[1].set_title(f'\n$R^2$={Rsq:.3}')
    xlim = [min(vals_all["activity_actual"]), max(vals_all["activity_actual"])]
    ax[1].plot(xlim, xlim, '--', color='k', label='diagonal', zorder=100)

    fig.suptitle("Validation Data")
    plt.savefig(f'{outdir}/validation_plots.png')
    plt.clf()

    # Scatter analysis - Training
    fig, ax = plt.subplots(1,2,figsize=[10,5])
    ax[0].scatter(train_all["abund_actual"],train_all["abund_pred"],color='C0', s=10, alpha=.3)
    Rsq = np.corrcoef(train_all["abund_actual"],train_all["abund_pred"])[0, 1]**2
    ax[0].set_xlabel('Measured Abundance')
    ax[0].set_ylabel('Predicted Abundance')
    ax[0].set_title(f'\n$R^2$={Rsq:.3}')
    xlim = [min(train_all["abund_actual"]), max(train_all["abund_actual"])]
    ax[0].plot(xlim, xlim, '--', color='k', label='diagonal', zorder=100)

    ax[1].scatter(train_all["activity_actual"],train_all["activity_pred"],color='C1', s=10, alpha=.3)
    Rsq = np.corrcoef(train_all["activity_actual"],train_all["activity_pred"])[0, 1]**2
    ax[1].set_xlabel('Measured Activity')
    ax[1].set_ylabel('Predicted Activity')
    ax[1].set_title(f'\n$R^2$={Rsq:.3}')
    xlim = [min(train_all["activity_actual"]), max(train_all["activity_actual"])]
    ax[1].plot(xlim, xlim, '--', color='k', label='diagonal', zorder=100)

    fig.suptitle("Training Data")
    plt.savefig(f'{outdir}/training_plots.png')
    plt.clf()

elif not simple_abund: 
    fig, ax = plt.subplots(1,2,figsize=[10,5])
    ax[0].scatter(train_all["activity_actual"],train_all["activity_pred"],color='C1', s=10, alpha=.3)
    Rsq = np.corrcoef(train_all["activity_actual"],train_all["activity_pred"])[0, 1]**2
    ax[0].set_xlabel('Measured Activity')
    ax[0].set_ylabel('Predicted Activity')
    ax[0].set_title(f'Training Data\n$R^2$={Rsq:.3}')
    xlim = [min(train_all["activity_actual"]), max(train_all["activity_actual"])]
    ax[0].plot(xlim, xlim, '--', color='k', label='diagonal', zorder=100)

    ax[1].scatter(vals_all["activity_actual"],vals_all["activity_pred"],color='C1', s=10, alpha=.3)
    Rsq = np.corrcoef(vals_all["activity_actual"],vals_all["activity_pred"])[0, 1]**2
    ax[1].set_xlabel('Measured Activity')
    ax[1].set_ylabel('Predicted Activity')
    ax[1].set_title(f'Validation Data\n$R^2$={Rsq:.3}')
    xlim = [min(vals_all["activity_actual"]), max(vals_all["activity_pred"])]
    ax[1].plot(xlim, xlim, '--', color='k', label='diagonal', zorder=100)

    plt.savefig(f'{outdir}/scatter_plots.png')
    plt.clf()
else: 
    fig, ax = plt.subplots(1,2,figsize=[10,5])
    ax[0].scatter(train_all["abund_actual"],train_all["abund_pred"],color='C0', s=10, alpha=.3)
    Rsq = np.corrcoef(train_all["abund_actual"],train_all["abund_pred"])[0, 1]**2
    ax[0].set_xlabel('Measured Abundance')
    ax[0].set_ylabel('Predicted Abundance')
    ax[0].set_title(f'Training Data\n$R^2$={Rsq:.3}')
    xlim = [min(train_all["abund_actual"]), max(train_all["abund_actual"])]
    ax[0].plot(xlim, xlim, '--', color='k', label='diagonal', zorder=100)

    ax[1].scatter(vals_all["abund_actual"],vals_all["abund_pred"],color='C0', s=10, alpha=.3)
    Rsq = np.corrcoef(vals_all["abund_actual"],vals_all["abund_pred"])[0, 1]**2
    ax[1].set_xlabel('Measured Abundance')
    ax[1].set_ylabel('Predicted Abundance')
    ax[1].set_title(f'Validation Data\n$R^2$={Rsq:.3}')
    xlim = [min(vals_all["abund_actual"]), max(vals_all["abund_pred"])]
    ax[1].plot(xlim, xlim, '--', color='k', label='diagonal', zorder=100)

    plt.savefig(f'{outdir}/scatter_plots.png')
    plt.clf()


# Plot losses
loss = pd.read_csv(f"{model}_losses.csv")
plt.plot(loss["epochs"],loss["train_loss"],label="Training data")
plt.plot(loss["epochs"],loss["valid_loss"], label="Validation data")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig(f'{outdir}/losses.png')
plt.clf()


