"""
Modified Date: 1 Feb 2025
Author: Claire LeBlanc

Usage: python Final_Model.py [optional args]
"""

import torch
from Data import DataReader, SplitData
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import torch.nn.init as init
import copy
import numpy as np
from scipy.stats import pearsonr
torch.manual_seed(25)

def one_hot_to_sequence(one_hot_tensor):
    """
    Convert a one-hot encoded tensor of protein sequences to the original sequences.

    Parameters:
    one_hot_tensor (torch.Tensor): A tensor of shape (num_sequences, sequence_length, num_amino_acids)
                                   representing one-hot encoded protein sequences.

    Returns:
    list of str: A list of the original protein sequences.
    """
    # Check if the tensor is on GPU and move it to CPU if necessary
    if one_hot_tensor.is_cuda:
        one_hot_tensor = one_hot_tensor.cpu()

    amino_acids = "RHKDESTNQCGPAVILMFYW"
    # Find the indices of the maximum values along the last dimension (num_amino_acids)
    indices = torch.argmax(one_hot_tensor, dim=-1)

    # Convert indices to amino acid characters
    sequences = []
    for seq_indices in indices:
        seq_indices = [item for sublist in seq_indices.tolist() for item in sublist]
        sequence = ''.join([amino_acids[i] for i in seq_indices])
        sequences.append(sequence)

    return sequences

class HillActivation(torch.nn.Module):
    """ 
    Implementation of the hill function used to go from bound transcription factor to activity.
    """
    def __init__(self, n=1):
        """
        Initalizes the class. Defines traininable parameters beta and K.
        """
        super(HillActivation, self).__init__()
        torch.manual_seed(25)

        # Define the three parameters we want the neural network too learn
        self.beta = torch.nn.Parameter(torch.Tensor([4.0])) #Beta is max expression value, starting this with max activity (i.e. fluorescence) in dataset
        self.K = torch.nn.Parameter(torch.Tensor([2.0])) #K is midpoint of hill curve, assuming no basal expression, half of beta
        self.n = float(n)  # Need this to be only integer values i.e. 1,2,4
        # self.relu = torch.nn.ReLU()

    def forward(self, x):
        """
        Applies hill function to tensor x.
        @return: New tensor with hill function applied
        """
        # x = self.relu(x) # Added this after all the models were trained
        xn = torch.pow(x, self.n) 
        kn = torch.pow(self.K, self.n)
        numerator = torch.mul(self.beta, xn)
        denom = torch.add(kn, xn)
        output = torch.div(numerator, denom)
        return output
    
    def reset_parameters(self):
        """
        Resets trainable parameters beta and K.
        """
        self.beta.data = torch.Tensor([4.0]) #Beta is max expression value, starting this with max activity (i.e. fluorescence) in dataset
        self.K.data = torch.Tensor([2.0]) #K is midpoint of hill curve, assuming no basal expression, half of beta

    def freeze_hill_params(self):
        self.beta.requires_grad = False
        self.K.requires_grad = False

        return
    
    def unfreeze_hill_params(self):
        self.beta.requires_grad = True
        self.K.requires_grad = True

        return

class ADModel_act(torch.nn.Module):
    """
    Same as ADModel except for instead of predicting closed, this predicts abundance directly 
    Uses a equlibrium model and a convolutional layer. 
    To use a single linear layer instead of a convolutional layer, set kernel_size to 40.
    """
    def __init__(self,input_shape,kernel_size=10, seed=25):
        """
        Initalizes the ADModel object.
        @param input_shape: The size of the input. 2D input is expected, so size should be a tuple of size 2 with (rows,columns)
        @param kernel_size: Size of the kernel i.e. how many adjacent amino acids should be considered at a time. 40 turns kernel into a single linear layer.
        """
        super(ADModel_act, self).__init__() # initalizing parent class (torch.nn.Module)
        torch.manual_seed(seed)
        
        self.input_shape = input_shape
        self.kernel_size = kernel_size

        # Parameters for convolutional layers
        out_channels = 1
        kernel_size = (kernel_size,20) # (sequence position, amino acid)
        conv_out_height = self.input_shape[1] - kernel_size[0] + 1
        conv_out_width = self.input_shape[2] - kernel_size[1] + 1
        linear_input_size = conv_out_height * conv_out_width * out_channels
        
        # Layers to predict k1
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=kernel_size)
        torch.nn.init.kaiming_normal_(self.conv1.weight, mode='fan_in', nonlinearity='leaky_relu') 
        torch.nn.init.constant_(self.conv1.bias, 0)

        self.linear1 = torch.nn.Linear(in_features=linear_input_size, out_features=1)
        torch.nn.init.kaiming_normal_(self.linear1.weight, mode='fan_in', nonlinearity='leaky_relu') 
        torch.nn.init.constant_(self.linear1.bias, 0)

        # Activation function 
        self.activate = torch.nn.PReLU(num_parameters=1, init=0.25)


    def forward(self,x): 
        """
        Applies the model to the input tensor to get output predictions. 
        @param x: The input tensor
        @return: A tensor containing abundance and activity predictions. 
        """
        x = x.float()

        # Estimating activity
        activitiy = self.conv1(x)
        activitiy = self.activate(activitiy)
        activitiy = activitiy.view(activitiy.size(0),activitiy.size(1)*activitiy.size(2)*activitiy.size(3)) # Linear layers only operate on 2d tensors
        activitiy = self.linear1(activitiy)
        activitiy = self.activate(activitiy)

        return activitiy
    
    
    def train_model(self,train_dataloader,val_dataloader,epochs,device,learning_rate,outfile,weight_pen=0):
        """
        Trains the model for a given number of epochs, stopping early if validation loss has not improved for five epochs.

        @param train_dataloader: Training data
        @param val_dataloader: Validation data
        @param epochs: Total number of epochs
        @param device: CPU or GPU depending on what's available
        @param learning_rate: Learning rate
        @param outfile: Name of saved model
        @return: The final loss
        """

        # Training parameters
        self.num_epochs = epochs
        self.outfile = outfile

        # Mean absolute error (average of sum of absolute differences)
        loss_function = torch.nn.L1Loss()

        optimizer = torch.optim.Adam(params=self.parameters(),lr=learning_rate)

        self.train_losses = []
        self.valid_losses = []

        # Initialize variables for early stopping
        best_metric = float('inf') 
        patience = 5  # Number of epochs with no improvement to wait before early stopping
        counter = 0  # Counter to keep track of epochs with no improvement

        for e in range(self.num_epochs):
            train_loss = 0.0
            
            # Set the model in training mode
            self.train()
            
            # Loop through training data, calculate loss and update weights
            for X,y in train_dataloader:
                X.to(device)
                y.to(device)
                
                X = X.to(torch.float32)
                
                # Caclulate the predicted values
                y_pred = self(X)
                weight_penalty = torch.sum(torch.nn.ReLU()(-self.linear1.weight))
                
                y = y.reshape([y.shape[0], y.shape[1], y.shape[2]])
                # ONLY TRAIN ON ABUND
                loss = loss_function(y_pred, y[:,1]) + weight_pen * weight_penalty

                optimizer.zero_grad()
                loss.backward() 
                optimizer.step()

                # Calculate loss
                train_loss += loss.item()

            self.train_losses.append(train_loss/len(train_dataloader))

            valid_loss = 0.0

            # Set the model to eval mode
            self.eval()    

            # Loop through the validation data and calculate loss (no weight updates)
            for X, y in val_dataloader:    
                y_pred = self(X)

                y = y.reshape([y.shape[0], y.shape[1], y.shape[2]])

                loss = loss_function(y_pred, y[:,1])
                valid_loss += loss.item()

            self.valid_losses.append(valid_loss/len(val_dataloader))

            # Check for early stopping
            if valid_loss < best_metric:
                best_metric = valid_loss
                counter = 0
                torch.save(self.state_dict(), f'{outfile}.pth')
            else:
                counter += 1

            if counter >= patience:
                print(f'Early stopping after {e+1} epochs')
                self.num_epochs = e+1
                break

            
            print(f'Epoch {e+1} \t\t Training Loss: {train_loss / len(train_dataloader)}')
            print(f'Epoch {e+1} \t\t Validation Loss: {valid_loss / len(val_dataloader)}') 

        return train_loss/len(train_dataloader), valid_loss/len(val_dataloader)

    
    def get_actual_and_predicted(self,dataloader,device,filepath, save_file=True):
        """
        Creates a CSV file with the actual and predicted values for each sequence.
        @param dataloader: Data to give to model
        @param device: Either CPU or GPU
        @param filepath: Where to save the results
        """
        print(self.outfile)
        all_data = pd.DataFrame([])

        best_model = self.__class__(self.input_shape,self.kernel_size)
        best_model.load_state_dict(torch.load(f'{self.outfile}.pth'))
        best_model.eval()

        for X,y in dataloader:
            X.to(device)
            y.to(device)
            
            X = X.to(torch.float32) # Added cause was getting error cause diff type than internal weights
            
            sequences = pd.DataFrame(one_hot_to_sequence(X))
            # Caclulate the predicted values
            y_pred = best_model(X)
            activity_pred = pd.DataFrame(y_pred.detach().numpy().flatten())
            activity_actual = pd.DataFrame(y[:,1].detach().numpy().flatten())

            data = pd.concat([activity_pred,activity_actual,sequences],axis=1)
            all_data = pd.concat([all_data, data])
        
        all_data.columns = ["activity_pred","activity_actual","aa_seq"]
        if save_file: 
            all_data.to_csv(filepath,index=False)
        

    def get_losses(self,outfile):
        """
        Plots the losses of each epoch and saves plot and values.
        @param outfile: Where to save the final figure and values. 
        """
        epochs = range(self.num_epochs)
        plt.plot(epochs,self.valid_losses,label="Test Loss")
        plt.plot(epochs,self.train_losses, label="Train Loss")
        plt.legend(loc="upper left")
        plt.xlabel("Epochs")
        plt.ylabel("Losses")
        # plt.show()
        plt.savefig(f'{outfile}.png')

        df = pd.DataFrame([range(self.num_epochs),self.valid_losses, self.train_losses])
        df = df.transpose()
        df.columns=['epochs','valid_loss','train_loss']

        df.to_csv(f'{outfile}_losses.csv')

class ADModel_abund(torch.nn.Module):
    """
    Same as ADModel except for instead of predicting closed, this predicts abundance directly 
    Uses a equlibrium model and a convolutional layer. 
    To use a single linear layer instead of a convolutional layer, set kernel_size to 40.
    """
    def __init__(self,input_shape,kernel_size=10, seed=25):
        """
        Initalizes the ADModel object.
        @param input_shape: The size of the input. 2D input is expected, so size should be a tuple of size 2 with (rows,columns)
        @param kernel_size: Size of the kernel i.e. how many adjacent amino acids should be considered at a time. 40 turns kernel into a single linear layer.
        """
        super(ADModel_abund, self).__init__() # initalizing parent class (torch.nn.Module)
        torch.manual_seed(seed)
        
        self.input_shape = input_shape
        self.kernel_size = kernel_size

        # Parameters for convolutional layers
        out_channels = 1
        kernel_size = (kernel_size,20) # (sequence position, amino acid)
        conv_out_height = self.input_shape[1] - kernel_size[0] + 1
        conv_out_width = self.input_shape[2] - kernel_size[1] + 1
        linear_input_size = conv_out_height * conv_out_width * out_channels
        
        # Layers to predict k1
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=kernel_size)
        torch.nn.init.kaiming_normal_(self.conv1.weight, mode='fan_in', nonlinearity='leaky_relu') 
        torch.nn.init.constant_(self.conv1.bias, 0)

        self.linear1 = torch.nn.Linear(in_features=linear_input_size, out_features=1)
        torch.nn.init.kaiming_normal_(self.linear1.weight, mode='fan_in', nonlinearity='leaky_relu') 
        # torch.nn.init.uniform_(self.linear1.weight, a=0.1, b=1.0) 
        # torch.nn.init.uniform_(self.linear1.bias, a=0.1, b=1.0) 
        # Initialize weights and biases with a constant value
        # torch.nn.init.constant_(self.linear1.weight, 0.5)  # All weights set to 0.5
        torch.nn.init.constant_(self.linear1.bias, 0)    # All biases set to 0.5


        # Activation function 
        self.activate = torch.nn.PReLU(num_parameters=1, init=0.25)


    def forward(self,x): 
        """
        Applies the model to the input tensor to get output predictions. 
        @param x: The input tensor
        @return: A tensor containing abundance and activity predictions. 
        """
        x = x.float()

        # Estimating activity
        activitiy = self.conv1(x)
        activitiy = self.activate(activitiy)
        activitiy = activitiy.view(activitiy.size(0),activitiy.size(1)*activitiy.size(2)*activitiy.size(3)) # Linear layers only operate on 2d tensors
        activitiy = self.linear1(activitiy)
        activitiy = self.activate(activitiy)

        return activitiy
    
    
    def train_model(self,train_dataloader,val_dataloader,epochs,device,learning_rate,outfile,weight_pen=0):
        """
        Trains the model for a given number of epochs, stopping early if validation loss has not improved for five epochs.

        @param train_dataloader: Training data
        @param val_dataloader: Validation data
        @param epochs: Total number of epochs
        @param device: CPU or GPU depending on what's available
        @param learning_rate: Learning rate
        @param outfile: Name of saved model
        @return: The final loss
        """

        # Training parameters
        self.num_epochs = epochs
        self.outfile = outfile

        # Mean absolute error (average of sum of absolute differences)
        loss_function = torch.nn.L1Loss()

        optimizer = torch.optim.Adam(params=self.parameters(),lr=learning_rate)

        self.train_losses = []
        self.valid_losses = []

        # Initialize variables for early stopping
        best_metric = float('inf') 
        patience = 5  # Number of epochs with no improvement to wait before early stopping
        counter = 0  # Counter to keep track of epochs with no improvement

        for e in range(self.num_epochs):
            train_loss = 0.0
            
            # Set the model in training mode
            self.train()
            
            # Loop through training data, calculate loss and update weights
            for X,y in train_dataloader:
                X.to(device)
                y.to(device)
                
                X = X.to(torch.float32)
                
                # Caclulate the predicted values
                y_pred = self(X)
                weight_penalty = torch.sum(torch.nn.ReLU()(-self.linear1.weight))
                
                y = y.reshape([y.shape[0], y.shape[1], y.shape[2]])
                # ONLY TRAIN ON ABUND
                loss = loss_function(y_pred, y[:,0]) + weight_pen * weight_penalty

                optimizer.zero_grad()
                loss.backward() 
                optimizer.step()

                # Calculate loss
                train_loss += loss.item()

            self.train_losses.append(train_loss/len(train_dataloader))

            valid_loss = 0.0

            # Set the model to eval mode
            self.eval()    

            # Loop through the validation data and calculate loss (no weight updates)
            for X, y in val_dataloader:    
                y_pred = self(X)

                y = y.reshape([y.shape[0], y.shape[1], y.shape[2]])

                loss = loss_function(y_pred, y[:,0])
                valid_loss += loss.item()

            self.valid_losses.append(valid_loss/len(val_dataloader))

            # Check for early stopping
            if valid_loss < best_metric:
                best_metric = valid_loss
                counter = 0
                torch.save(self.state_dict(), f'{outfile}.pth')
            else:
                counter += 1

            if counter >= patience:
                print(f'Early stopping after {e+1} epochs')
                self.num_epochs = e+1
                break

            
            print(f'Epoch {e+1} \t\t Training Loss: {train_loss / len(train_dataloader)}')
            print(f'Epoch {e+1} \t\t Validation Loss: {valid_loss / len(val_dataloader)}') 

        return train_loss/len(train_dataloader), valid_loss/len(val_dataloader)

    
    def get_actual_and_predicted(self,dataloader,device,filepath, save_file=True):
        """
        Creates a CSV file with the actual and predicted values for each sequence.
        @param dataloader: Data to give to model
        @param device: Either CPU or GPU
        @param filepath: Where to save the results
        """
        print(self.outfile)
        all_data = pd.DataFrame([])

        best_model = self.__class__(self.input_shape,self.kernel_size)
        best_model.load_state_dict(torch.load(f'{self.outfile}.pth'))
        best_model.eval()

        for X,y in dataloader:
            X.to(device)
            y.to(device)
            
            X = X.to(torch.float32) # Added cause was getting error cause diff type than internal weights
            
            sequences = pd.DataFrame(one_hot_to_sequence(X))
            # Caclulate the predicted values
            y_pred = best_model(X)
            abund_pred = pd.DataFrame(y_pred.detach().numpy().flatten())
            abund_actual = pd.DataFrame(y[:,0].detach().numpy().flatten())

            data = pd.concat([abund_pred,abund_actual,sequences],axis=1)
            all_data = pd.concat([all_data, data])
        
        all_data.columns = ["abund_pred","abund_actual","aa_seq"]
        if save_file: 
            all_data.to_csv(filepath,index=False)
        

    def get_losses(self,outfile):
        """
        Plots the losses of each epoch and saves plot and values.
        @param outfile: Where to save the final figure and values. 
        """
        epochs = range(self.num_epochs)
        plt.plot(epochs,self.valid_losses,label="Test Loss")
        plt.plot(epochs,self.train_losses, label="Train Loss")
        plt.legend(loc="upper left")
        plt.xlabel("Epochs")
        plt.ylabel("Losses")
        # plt.show()
        plt.savefig(f'{outfile}.png')

        df = pd.DataFrame([range(self.num_epochs),self.valid_losses, self.train_losses])
        df = df.transpose()
        df.columns=['epochs','valid_loss','train_loss']

        df.to_csv(f'{outfile}_losses.csv')

class ADModel_three_state(torch.nn.Module):
    """
    Same as ADModel except for instead of predicting closed, this predicts abundance directly 
    Uses a equlibrium model and a convolutional layer. 
    To use a single linear layer instead of a convolutional layer, set kernel_size to 40.
    """
    def __init__(self,input_shape,transformation,kernel_size=10,outchannel=1,relu = True,positive=False, hill_val=1, seed=25):
        """
        Initalizes the ADModel object.
        @param input_shape: The size of the input. 2D input is expected, so size should be a tuple of size 2 with (rows,columns)
        @param transformation: Function to go from bound to activity. One of the following options: 'Linear', 'Hill' or 'Exponential.' 
        @param kernel_size: Size of the kernel i.e. how many adjacent amino acids should be considered at a time. 40 turns kernel into a single linear layer.
        @param outchannel: How many different kernels to use. Outchannel of 1 is most interpretable but slightly less good.
        @param relu: True indicates to use the ReLU activation function, while False indicates using the Parametric ReLU function.
        """
        
        super(ADModel_three_state, self).__init__() # initalizing parent class (torch.nn.Module)
        torch.manual_seed(seed)

        self.input_shape = input_shape
        self.relu = relu
        self.positive = positive
        self.kernel_size = kernel_size
        self.outchannel = outchannel
        self.positive = positive
        self.hill_val = hill_val
        self.transformation = transformation

        # Parameters for convolutional layers
        kernel_size = (kernel_size,20) # (sequence position, amino acid)
        conv_out_height =self.input_shape[1] - kernel_size[0] + 1
        conv_out_width = self.input_shape[2] - kernel_size[1] + 1
        linear_input_size = conv_out_height * conv_out_width * self.outchannel
        
        # Layers to predict abundance
        self.conv3 = torch.nn.Conv2d(in_channels=1, out_channels=self.outchannel, kernel_size=kernel_size)
        torch.nn.init.kaiming_normal_(self.conv3.weight, mode='fan_in', nonlinearity='leaky_relu') 

        self.linear3 = torch.nn.Linear(in_features=linear_input_size, out_features=1)
        torch.nn.init.kaiming_normal_(self.linear3.weight, mode='fan_in', nonlinearity='leaky_relu') 

        # Layers to predict k1
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=self.outchannel, kernel_size=kernel_size)
        torch.nn.init.kaiming_normal_(self.conv1.weight, mode='fan_in', nonlinearity='leaky_relu') 

        self.linear1 = torch.nn.Linear(in_features=linear_input_size, out_features=1)
        torch.nn.init.kaiming_normal_(self.linear1.weight, mode='fan_in', nonlinearity='leaky_relu') 

        # Layers to predict k2
        self.conv2 = torch.nn.Conv2d(in_channels=1, out_channels=self.outchannel, kernel_size=kernel_size)
        torch.nn.init.kaiming_normal_(self.conv2.weight, mode='fan_in', nonlinearity='leaky_relu') 

        self.linear2 = torch.nn.Linear(in_features=linear_input_size, out_features=1)
        torch.nn.init.kaiming_normal_(self.linear2.weight, mode='fan_in', nonlinearity='leaky_relu') 


        # Shared activation function
        if self.relu:    
            self.activate = torch.nn.ReLU()
        else:
            self.activate1 = torch.nn.PReLU(num_parameters=1, init=0.25)
            self.activate2 = torch.nn.PReLU(num_parameters=1, init=0.25)
            self.activate3 = torch.nn.PReLU(num_parameters=1, init=0.25)
            self.activate4 = torch.nn.PReLU(num_parameters=1, init=0.25)
            self.activate5 = torch.nn.PReLU(num_parameters=1, init=0.25)
            self.activate6 = torch.nn.PReLU(num_parameters=1, init=0.25)

        self.sigmoid = torch.nn.Sigmoid()
        
        # Layer to go from bound to activity
        if transformation == "Linear":
            self.islinear = True
            self.transf = torch.nn.Linear(1, 1)
            self.transf.reset_parameters() 
        elif transformation == "Hill":
            self.islinear = False
            self.transf = HillActivation(hill_val)
            self.transf.reset_parameters()
        else:
            self.islinear = True
            print("Not a valid transformation, defaulting to Linear")
            self.transf = torch.nn.Linear(1, 1)

    def forward(self,x): 
        """
        Applies the model to the input tensor to get output predictions. 
        @param x: The input tensor
        @return: A tensor containing abundance and activity predictions. 
        """
        x = x.float()

        # Estimating bound
        abund = self.conv3(x)
        abund = self.activate1(abund)
        abund = abund.view(abund.size(0),abund.size(1)*abund.size(2)*abund.size(3)) # Linear layers only operate on 2d tensors
        abund = self.linear3(abund)
        if not self.relu:
            abund = self.activate2(abund)
        # abund = self.sigmoid(abund)

        # Estimating k1
        k1 = self.conv1(x)
        k1 = self.activate3(k1)
        k1 = k1.view(k1.size(0),k1.size(1)*k1.size(2)*k1.size(3)) # Linear layers only operate on 2d tensors
        k1 = self.linear1(k1)
        if not self.relu: 
            k1 = self.activate4(k1)

        # Estimating k2
        k2 = self.conv2(x)
        k2 = self.activate5(k2)
        k2 = k2.view(k2.size(0),k2.size(1)*k2.size(2)*k2.size(3)) # Linear layers only operate on 2d tensors
        k2 = self.linear2(k2)
        if not self.relu:
            k2 = self.activate6(k2)

        # Saving Ks for future access
        self.K1 = k1
        self.K2 = k2 
        self.abund = abund

        # bound = torch.div(abund,torch.add(1,torch.add(torch.div(1,torch.mul(k1,k2)),torch.div(1,k2))))
        bound = torch.div(abund,torch.add(1,torch.mul(torch.div(1,k2),torch.add(1,torch.div(1,k1)))))
        activity = self.transf(bound)
        # abund = k1
        # activity = k2
        return torch.stack((abund, activity)).transpose(0,1)
    
    def get_ks(self):
        """
        Run directly after calling forward on tensor of interest. 
        @return: The K1, K2, and closed values for the input tensor x. 
        """
        return self.K1, self.K2, self.abund
    
    def train_model_abund(self,train_dataloader,val_dataloader,epochs,device,learning_rate,outfile, neg_pen, weight_pen, act_pen):
        """
        Trains the model for a given number of epochs, stopping early if validation loss has not improved for five epochs.

        @param train_dataloader: Training data
        @param val_dataloader: Validation data
        @param epochs: Total number of epochs
        @param device: CPU or GPU depending on what's available
        @param learning_rate: Learning rate
        @param outfile: Name of saved model
        @return: The final loss
        """

        if not self.islinear:
            # Do not learn the hill function right now
            self.transf.freeze_hill_params()

        # Training parameters
        self.num_epochs = epochs
        self.outfile = outfile

        # Mean absolute error (average of sum of absolute differences)
        loss_function = torch.nn.L1Loss()
        # loss_function = torch.nn.MSELoss()

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()),lr=learning_rate)
        # optimizer = torch.optim.Adam(params=self.parameters(),lr=learning_rate)

        self.train_losses = []
        self.valid_losses = []

        # Initialize variables for early stopping
        best_metric = float('inf') 
        patience = 5  # Number of epochs with no improvement to wait before early stopping
        counter = 0  # Counter to keep track of epochs with no improvement

        # Initalize loss weight values
        activity_weight = act_pen
        abundance_weight = 2 - activity_weight

        for e in range(self.num_epochs):
            train_loss = 0.0
            
            # Set the model in training mode
            self.train()
            
            # Loop through training data, calculate loss and update weights
            for X,y in train_dataloader:
                X.to(device)
                y.to(device)
                
                X = X.to(torch.float32)
                
                # Caclulate the predicted values
                y_pred = self(X)

                k1, k2, abund = self.get_ks()
                negative_penalty = torch.sum(torch.nn.ReLU()(-abund))
                weight_penalty = torch.sum(torch.nn.ReLU()(-self.linear3.weight)) #+ torch.nn.ReLU()(-self.transf.weight))

                y = y.reshape([y.shape[0], y.shape[1], y.shape[2]])

                # ONLY TRAIN ON ABUND
                loss = loss_function(y_pred[:,0], y[:,0]) + neg_pen * negative_penalty + weight_pen * weight_penalty 
                # loss = negative_penalty + weight_penalty

                optimizer.zero_grad()
                loss.backward() 
                optimizer.step()

                # Calculate loss
                train_loss += loss.item()

            self.train_losses.append(train_loss/len(train_dataloader))

            valid_loss = 0.0

            # Set the model to eval mode
            self.eval()    

            # Loop through the validation data and calculate loss (no weight updates)
            for X, y in val_dataloader:    
                y_pred = self(X)

                k1, k2, abund = self.get_ks()
                negative_penalty = torch.sum(torch.nn.ReLU()(-abund))
                weight_penalty = torch.sum(torch.nn.ReLU()(-self.linear3.weight)) #+ torch.nn.ReLU()(-self.transf.weight))

                y = y.reshape([y.shape[0], y.shape[1], y.shape[2]])

                # ONLY TRAINING ON ABUNDANCE LOSS
                loss = loss_function(y_pred[:,0], y[:,0]) + neg_pen * negative_penalty + weight_pen * weight_penalty 
                valid_loss += loss.item()

            self.valid_losses.append(valid_loss/len(val_dataloader))

            # Check for early stopping
            if valid_loss < best_metric:
                best_metric = valid_loss
                counter = 0
                torch.save(self.state_dict(), f'{outfile}.pth')
            else:
                counter += 1

            if counter >= patience:
                print(f'Early stopping after {e+1} epochs')
                self.num_epochs = e+1
                break

            
            print(f'Epoch {e+1} \t\t Training Loss: {train_loss / len(train_dataloader)}')
            print(f'Epoch {e+1} \t\t Validation Loss: {valid_loss / len(val_dataloader)}') 
            # print(f'Epoch {e+1} \t\t Negative Penalty: {negative_penalty}') 
            # print(self.transf.weight)

        return train_loss/len(train_dataloader), valid_loss/len(val_dataloader)

    def freeze_abund_weights(self):
        for param in self.conv3.parameters():
            param.requires_grad = False

        for param in self.activate1.parameters():
            param.requires_grad = False

        for param in self.linear3.parameters():
            param.requires_grad = False

        for param in self.activate2.parameters():
            param.requires_grad = False

        return
    
    def unfreeze_act_weights(self):
        for param in self.conv1.parameters():
            param.requires_grad = True

        for param in self.activate3.parameters():
            param.requires_grad = True

        for param in self.linear1.parameters():
            param.requires_grad = True

        for param in self.activate4.parameters():
            param.requires_grad = True

        for param in self.conv2.parameters():
            param.requires_grad = True

        for param in self.activate5.parameters():
            param.requires_grad = True

        for param in self.linear2.parameters():
            param.requires_grad = True

        for param in self.activate6.parameters():
            param.requires_grad = True

        return

    def freeze_act_weights(self):
        for param in self.conv1.parameters():
            param.requires_grad = False

        for param in self.activate3.parameters():
            param.requires_grad = False

        for param in self.linear1.parameters():
            param.requires_grad = False

        for param in self.activate4.parameters():
            param.requires_grad = False

        for param in self.conv2.parameters():
            param.requires_grad = False

        for param in self.activate5.parameters():
            param.requires_grad = False

        for param in self.linear2.parameters():
            param.requires_grad = False

        for param in self.activate6.parameters():
            param.requires_grad = False

        return
    

    def train_model_act(self,train_dataloader,val_dataloader,epochs,device,learning_rate,outfile,neg_pen, weight_pen, act_pen):
        """
        Trains the model for a given number of epochs, stopping early if validation loss has not improved for five epochs.

        @param train_dataloader: Training data
        @param val_dataloader: Validation data
        @param epochs: Total number of epochs
        @param device: CPU or GPU depending on what's available
        @param learning_rate: Learning rate
        @param outfile: Name of saved model
        @return: The final loss
        """
        if not self.islinear:
            # Learn the hill function right now
            self.transf.unfreeze_hill_params()

        # Training parameters
        self.num_epochs = epochs
        self.outfile = outfile

        # Mean absolute error (average of sum of absolute differences)
        loss_function = torch.nn.L1Loss()

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()),lr=learning_rate)

        self.train_losses = []
        self.valid_losses = []

        # Initialize variables for early stopping
        best_metric = float('inf') 
        patience = 5  # Number of epochs with no improvement to wait before early stopping
        counter = 0  # Counter to keep track of epochs with no improvement

        # Initalize loss weight values
        activity_weight = act_pen
        abundance_weight = 2 - activity_weight

        for e in range(self.num_epochs):
            train_loss = 0.0
            
            # Set the model in training mode
            self.train()
            
            # Loop through training data, calculate loss and update weights
            for X,y in train_dataloader:
                X.to(device)
                y.to(device)
                
                X = X.to(torch.float32)
                
                # Caclulate the predicted values
                y_pred = self(X)

                k1, k2, closed = self.get_ks()
                negative_penalty = torch.sum(torch.nn.ReLU()(-k1) + torch.nn.ReLU()(-k2))
                weight_penalty = torch.sum(torch.nn.ReLU()(-self.linear1.weight)) + torch.sum(torch.nn.ReLU()(-self.linear2.weight))

                y = y.reshape([y.shape[0], y.shape[1], y.shape[2]])
                
                # Calculate the difference between the predicted values and the actual values
                # ONLY TRAIN ON ACTIVITY
                loss = loss_function(y_pred[:,1], y[:,1]) + neg_pen * negative_penalty + weight_pen * weight_penalty                # loss = negative_penalty + weight_penalty

                optimizer.zero_grad()
                loss.backward() 
                optimizer.step()

                # Calculate loss
                train_loss += loss.item()

            self.train_losses.append(train_loss/len(train_dataloader))

            valid_loss = 0.0

            # Set the model to eval mode
            self.eval()    

            # Loop through the validation data and calculate loss (no weight updates)
            for X, y in val_dataloader:    
                y_pred = self(X)
                k1, k2, closed = self.get_ks()
                negative_penalty = torch.sum(torch.nn.ReLU()(-k1) + torch.nn.ReLU()(-k2))
                weight_penalty = torch.sum(torch.nn.ReLU()(-self.linear1.weight)) + torch.sum(torch.nn.ReLU()(-self.linear2.weight))

                y = y.reshape([y.shape[0], y.shape[1], y.shape[2]])

                # ONLY TRAIN ON ACTIVITY
                loss = loss_function(y_pred[:,1], y[:,1]) + neg_pen * negative_penalty + weight_pen * weight_penalty
                # loss = negative_penalty + weight_penalty
                valid_loss += loss.item()

            self.valid_losses.append(valid_loss/len(val_dataloader))

                        # Check for early stopping
            if valid_loss < best_metric:
                best_metric = valid_loss
                counter = 0
                torch.save(self.state_dict(), f'{outfile}.pth')
            else:
                counter += 1

            if counter >= patience:
                print(f'Early stopping after {e+1} epochs')
                self.num_epochs = e+1
                break

            print(f'Epoch {e+1} \t\t Training Loss: {train_loss / len(train_dataloader)}')
            print(f'Epoch {e+1} \t\t Validation Loss: {valid_loss / len(val_dataloader)}') 
            # print(f'Epoch {e+1} \t\t Negative Penalty: {negative_penalty}') 
    
    def get_actual_and_predicted(self,dataloader,device,filepath, save_file=True):
        """
        Creates a CSV file with the actual and predicted values for each sequence.
        @param dataloader: Data to give to model
        @param device: Either CPU or GPU
        @param filepath: Where to save the results
        """
        print(f"Saving predictions to {self.outfile}")
        all_data = pd.DataFrame([])
        best_model = self.__class__(self.input_shape,self.transformation,self.kernel_size,self.outchannel,self.relu,self.positive, self.hill_val)
        best_model.load_state_dict(torch.load(f'{self.outfile}.pth'))
        best_model.eval()

        for X,y in dataloader:
            X.to(device)
            y.to(device)
            
            X = X.to(torch.float32) # Added cause was getting error cause diff type than internal weights
            
            sequences = pd.DataFrame(one_hot_to_sequence(X))
            # Caclulate the predicted values
            y_pred = best_model(X)
            abund_pred = pd.DataFrame(y_pred[:,0].detach().numpy().flatten())
            activity_pred = pd.DataFrame(y_pred[:,1].detach().numpy().flatten())
            abund_actual = pd.DataFrame(y[:,0].detach().numpy().flatten())
            activity_actual = pd.DataFrame(y[:,1].detach().numpy().flatten())

            data = pd.concat([abund_pred,abund_actual,activity_pred,activity_actual,sequences],axis=1)
            all_data = pd.concat([all_data, data])
        
        all_data.columns = ["abund_pred","abund_actual","activity_pred","activity_actual","aa_seq"]
        if save_file: 
            all_data.to_csv(filepath,index=False)
        
        pearson_abund = pearsonr(all_data["abund_actual"],all_data["abund_pred"]).correlation


    def get_losses(self,outfile):
        """
        Plots the losses of each epoch and saves plot and values.
        @param outfile: Where to save the final figure and values. 
        """
        epochs = range(self.num_epochs)
        plt.plot(epochs,self.valid_losses,label="Test Loss")
        plt.plot(epochs,self.train_losses, label="Train Loss")
        plt.legend(loc="upper left")
        plt.xlabel("Epochs")
        plt.ylabel("Losses")
        # plt.show()
        plt.savefig(f'{outfile}.png')

        df = pd.DataFrame([range(self.num_epochs),self.valid_losses, self.train_losses])
        df = df.transpose()
        df.columns=['epochs','valid_loss','train_loss']

        df.to_csv(f'{outfile}_losses.csv')

class ADModel_three_state_abund(ADModel_three_state):
    """
    Same as ADModel except for instead of predicting closed, this predicts abundance directly 
    Uses a equlibrium model and a convolutional layer. 
    To use a single linear layer instead of a convolutional layer, set kernel_size to 40.
    """
    def __init__(self,input_shape,transformation,kernel_size=10,outchannel=1,relu = True,positive=False, hill_val=1, abund_k=15, seed=25):
        """
        Initalizes the ADModel object.
        @param input_shape: The size of the input. 2D input is expected, so size should be a tuple of size 2 with (rows,columns)
        @param transformation: Function to go from bound to activity. One of the following options: 'Linear', 'Hill' or 'Exponential.' 
        @param kernel_size: Size of the kernel i.e. how many adjacent amino acids should be considered at a time. 40 turns kernel into a single linear layer.
        @param outchannel: How many different kernels to use. Outchannel of 1 is most interpretable but slightly less good.
        @param relu: True indicates to use the ReLU activation function, while False indicates using the Parametric ReLU function.
        """
        super(ADModel_three_state_abund, self).__init__(input_shape, transformation, kernel_size, outchannel, relu, positive, hill_val) # initalizing parent class (torch.nn.Module)
        torch.manual_seed(seed)

        self.input_shape = input_shape
        self.relu = relu
        self.positive = positive
        self.kernel_size = kernel_size
        self.outchannel = outchannel
        self.positive = positive
        self.hill_val = hill_val
        self.transformation = transformation
        self.abund_k = abund_k

        
        # Layers to predict abundance
        self.conv3 = torch.nn.Conv2d(in_channels=1, out_channels=self.outchannel, kernel_size=(abund_k,20))
        torch.nn.init.kaiming_normal_(self.conv3.weight, mode='fan_in', nonlinearity='leaky_relu') 
        torch.nn.init.constant_(self.conv3.bias, 0)
    

        conv_out_height =self.input_shape[1] - abund_k + 1
        conv_out_width = self.input_shape[2] - 20 + 1
        linear_input_size = conv_out_height * conv_out_width * self.outchannel
        self.linear3 = torch.nn.Linear(in_features=linear_input_size, out_features=1)
        torch.nn.init.kaiming_normal_(self.linear3.weight, mode='fan_in', nonlinearity='leaky_relu') 
        torch.nn.init.constant_(self.linear3.bias, 0) 

        # Parameters for convolutional layers
        kernel_size = (kernel_size,20) # (sequence position, amino acid)
        conv_out_height =self.input_shape[1] - kernel_size[0] + 1
        conv_out_width = self.input_shape[2] - kernel_size[1] + 1
        linear_input_size = conv_out_height * conv_out_width * self.outchannel

        # Layers to predict k1
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=self.outchannel, kernel_size=kernel_size)
        torch.nn.init.kaiming_normal_(self.conv1.weight, mode='fan_in', nonlinearity='leaky_relu') 
        torch.nn.init.constant_(self.conv1.bias, 0)

        self.linear1 = torch.nn.Linear(in_features=linear_input_size, out_features=1)
        torch.nn.init.kaiming_normal_(self.linear1.weight, mode='fan_in', nonlinearity='leaky_relu') 
        torch.nn.init.constant_(self.linear1.bias, 0)

        # Layers to predict k2
        self.conv2 = torch.nn.Conv2d(in_channels=1, out_channels=self.outchannel, kernel_size=kernel_size)
        torch.nn.init.kaiming_normal_(self.conv2.weight, mode='fan_in', nonlinearity='leaky_relu') 
        torch.nn.init.constant_(self.conv2.bias, 0)

        self.linear2 = torch.nn.Linear(in_features=linear_input_size, out_features=1)
        torch.nn.init.kaiming_normal_(self.linear2.weight, mode='fan_in', nonlinearity='leaky_relu') 
        torch.nn.init.constant_(self.linear2.bias, 0)

        # Shared activation function
        if self.relu:    
            self.activate = torch.nn.ReLU()
        else:
            self.activate1 = torch.nn.PReLU(num_parameters=1, init=0.25)
            self.activate2 = torch.nn.PReLU(num_parameters=1, init=0.25)
            self.activate3 = torch.nn.PReLU(num_parameters=1, init=0.25)
            self.activate4 = torch.nn.PReLU(num_parameters=1, init=0.25)
            self.activate5 = torch.nn.PReLU(num_parameters=1, init=0.25)
            self.activate6 = torch.nn.PReLU(num_parameters=1, init=0.25)

        self.sigmoid = torch.nn.Sigmoid()
        
        # Layer to go from bound to activity
        if transformation == "Linear":
            self.transf = torch.nn.Linear(1, 1)
            self.transf.reset_parameters() 
        elif transformation == "Hill":
            self.transf = HillActivation(hill_val)
            self.transf.reset_parameters()
        else:
            print("Not a valid transformation, defaulting to Linear")
            self.transf = torch.nn.Linear(1, 1)

    def get_actual_and_predicted(self,dataloader,device,filepath, save_file=True):
        """
        Creates a CSV file with the actual and predicted values for each sequence.
        @param dataloader: Data to give to model
        @param device: Either CPU or GPU
        @param filepath: Where to save the results
        """
        print(f"Saving predictions to {self.outfile}")
        all_data = pd.DataFrame([])

        best_model = self.__class__(self.input_shape,self.transformation,self.kernel_size,self.outchannel,self.relu,self.positive, self.hill_val, self.abund_k)
        best_model.load_state_dict(torch.load(f'{self.outfile}.pth'))
        best_model.eval()

        for X,y in dataloader:
            X.to(device)
            y.to(device)
            
            X = X.to(torch.float32) # Added cause was getting error cause diff type than internal weights
            
            sequences = pd.DataFrame(one_hot_to_sequence(X))
            # Caclulate the predicted values
            y_pred = best_model(X)
            abund_pred = pd.DataFrame(y_pred[:,0].detach().numpy().flatten())
            activity_pred = pd.DataFrame(y_pred[:,1].detach().numpy().flatten())
            abund_actual = pd.DataFrame(y[:,0].detach().numpy().flatten())
            activity_actual = pd.DataFrame(y[:,1].detach().numpy().flatten())

            data = pd.concat([abund_pred,abund_actual,activity_pred,activity_actual,sequences],axis=1)
            all_data = pd.concat([all_data, data])

        all_data.columns = ["abund_pred","abund_actual","activity_pred","activity_actual","aa_seq"]
        if save_file: 
            all_data.to_csv(filepath,index=False)

        pearson_abund = pearsonr(all_data["abund_actual"],all_data["abund_pred"]).correlation

class ADModel_two_state(torch.nn.Module):
    """
    Same as ADModel except for instead of predicting closed, this predicts abundance directly 
    Uses a equlibrium model and a convolutional layer. 
    To use a single linear layer instead of a convolutional layer, set kernel_size to 40.
    """
    def __init__(self,input_shape,transformation,kernel_size=10,outchannel=1,relu = True,positive=False, hill_val=1, seed=25):
        """
        Initalizes the ADModel object.
        @param input_shape: The size of the input. 2D input is expected, so size should be a tuple of size 2 with (rows,columns)
        @param transformation: Function to go from bound to activity. One of the following options: 'Linear', 'Hill' or 'Exponential.' 
        @param kernel_size: Size of the kernel i.e. how many adjacent amino acids should be considered at a time. 40 turns kernel into a single linear layer.
        @param outchannel: How many different kernels to use. Outchannel of 1 is most interpretable but slightly less good.
        @param relu: True indicates to use the ReLU activation function, while False indicates using the Parametric ReLU function.
        """
        super(ADModel_two_state, self).__init__() # initalizing parent class (torch.nn.Module)
        torch.manual_seed(seed)

        self.input_shape = input_shape
        self.relu = relu
        self.positive = positive
        self.transformation = transformation
        self.kernel_size = kernel_size
        self.outchannel = outchannel
        self.hill_val = hill_val

        # Parameters for convolutional layers
        kernel_size = (kernel_size,20) # (sequence position, amino acid)
        conv_out_height =self.input_shape[1] - kernel_size[0] + 1
        conv_out_width = self.input_shape[2] - kernel_size[1] + 1
        linear_input_size = conv_out_height * conv_out_width * self.outchannel
        
        # Layers to predict inactive
        self.conv2 = torch.nn.Conv2d(in_channels=1, out_channels=self.outchannel, kernel_size=kernel_size)
        torch.nn.init.kaiming_normal_(self.conv2.weight, mode='fan_in', nonlinearity='leaky_relu')
        torch.nn.init.constant_(self.conv2.bias, 0) 

        linear_input_size = conv_out_height * conv_out_width * self.outchannel
        self.linear2 = torch.nn.Linear(in_features=linear_input_size, out_features=1)
        torch.nn.init.kaiming_normal_(self.linear2.weight, mode='fan_in', nonlinearity='leaky_relu') 
        torch.nn.init.constant_(self.linear2.bias, 0) 

        # Layers to predict K
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=self.outchannel, kernel_size=kernel_size)
        torch.nn.init.kaiming_normal_(self.conv1.weight, mode='fan_in', nonlinearity='leaky_relu') 
        torch.nn.init.constant_(self.conv1.bias, 0) 

        self.linear1 = torch.nn.Linear(in_features=linear_input_size, out_features=1)
        torch.nn.init.kaiming_normal_(self.linear1.weight, mode='fan_in', nonlinearity='leaky_relu') 
        torch.nn.init.constant_(self.linear1.bias, 0) 

        # Shared activation function
        if self.relu:    
            self.activate1 = torch.nn.ReLU()
            self.activate3 = torch.nn.ReLU()
        else:
            self.activate1 = torch.nn.PReLU(num_parameters=1, init=0.25)
            self.activate2 = torch.nn.PReLU(num_parameters=1, init=0.25)
            self.activate3 = torch.nn.PReLU(num_parameters=1, init=0.25)
            self.activate4 = torch.nn.PReLU(num_parameters=1, init=0.25)

        self.sigmoid = torch.nn.Sigmoid()
        
        # Layer to go from bound to activity
        if transformation == "Linear":
            self.transf = torch.nn.Linear(1, 1)
            self.transf.reset_parameters() 
        elif transformation == "Hill":
            self.transf = HillActivation(hill_val)
            self.transf.reset_parameters()
        else:
            print("Not a valid transformation, defaulting to Linear")
            self.transf = torch.nn.Linear(1, 1)

    def forward(self,x): 
        """
        Applies the model to the input tensor to get output predictions. 
        @param x: The input tensor
        @return: A tensor containing abundance and activity predictions. 
        """
        x = x.float()

        # Estimating K
        K = self.conv1(x)
        K = self.activate1(K)
        K = K.view(K.size(0),K.size(1)*K.size(2)*K.size(3)) # Linear layers only operate on 2d tensors
        K = self.linear1(K)
        if not self.relu: 
            K = self.activate2(K)

        # Estimating inactive
        abund = self.conv2(x)
        abund = self.activate3(abund)
        abund = abund.view(abund.size(0),abund.size(1)*abund.size(2)*abund.size(3)) # Linear layers only operate on 2d tensors
        abund = self.linear2(abund)
        if not self.relu:
            abund = self.activate4(abund)

        # Saving Ks for future access
        self.K = K
        self.abund = abund
        
        bound = torch.div(abund, torch.add(1, torch.div(1,K)))
        activity = self.transf(bound)
        # activity = k2
        return torch.stack((abund, activity)).transpose(0,1)
    
    def get_ks(self):
        """
        Run directly after calling forward on tensor of interest. 
        @return: The K1, K2, and closed values for the input tensor x. 
        """
        return self.K, self.abund

    def train_model_abund(self,train_dataloader,val_dataloader,epochs,device,learning_rate,outfile, neg_pen, weight_pen, act_pen):
        """
        Trains the model for a given number of epochs, stopping early if validation loss has not improved for five epochs.

        @param train_dataloader: Training data
        @param val_dataloader: Validation data
        @param epochs: Total number of epochs
        @param device: CPU or GPU depending on what's available
        @param learning_rate: Learning rate
        @param outfile: Name of saved model
        @return: The final loss
        """

        # Training parameters
        self.num_epochs = epochs
        self.outfile = outfile

        # Mean absolute error (average of sum of absolute differences)
        loss_function = torch.nn.L1Loss()
        # loss_function = torch.nn.MSELoss()

        optimizer = torch.optim.Adam(params=self.parameters(),lr=learning_rate)

        self.train_losses = []
        self.valid_losses = []

        # Initialize variables for early stopping
        best_metric = float('inf') 
        patience = 5  # Number of epochs with no improvement to wait before early stopping
        counter = 0  # Counter to keep track of epochs with no improvement

        # Initalize loss weight values
        activity_weight = act_pen
        abundance_weight = 2 - activity_weight

        for e in range(self.num_epochs):
            train_loss = 0.0
            
            # Set the model in training mode
            self.train()
            
            # Loop through training data, calculate loss and update weights
            for X,y in train_dataloader:
                X.to(device)
                y.to(device)
                
                X = X.to(torch.float32)
                
                # Caclulate the predicted values
                y_pred = self(X)

                k1, abund = self.get_ks()
                negative_penalty = torch.sum(torch.nn.ReLU()(-abund))
                weight_penalty = torch.sum(torch.nn.ReLU()(-self.linear2.weight)) #+ torch.nn.ReLU()(-self.transf.weight))

                y = y.reshape([y.shape[0], y.shape[1], y.shape[2]])

                # ONLY TRAIN ON ABUND
                loss = loss_function(y_pred[:,0], y[:,0]) + neg_pen * negative_penalty + weight_pen * weight_penalty 
                # loss = negative_penalty + weight_penalty

                optimizer.zero_grad()
                loss.backward() 
                optimizer.step()

                # Calculate loss
                train_loss += loss.item()

            self.train_losses.append(train_loss/len(train_dataloader))

            valid_loss = 0.0

            # Set the model to eval mode
            self.eval()    

            # Loop through the validation data and calculate loss (no weight updates)
            for X, y in val_dataloader:    
                y_pred = self(X)

                k1, abund = self.get_ks()
                negative_penalty = torch.sum(torch.nn.ReLU()(-abund))
                weight_penalty = torch.sum(torch.nn.ReLU()(-self.linear2.weight)) #+ torch.nn.ReLU()(-self.transf.weight))

                y = y.reshape([y.shape[0], y.shape[1], y.shape[2]])

                # ONLY TRAINING ON ABUNDANCE LOSS
                loss = loss_function(y_pred[:,0], y[:,0]) + neg_pen * negative_penalty + weight_pen * weight_penalty 
                valid_loss += loss.item()

            self.valid_losses.append(valid_loss/len(val_dataloader))

            # Check for early stopping
            if valid_loss < best_metric:
                best_metric = valid_loss
                counter = 0
                torch.save(self.state_dict(), f'{outfile}.pth')
            else:
                counter += 1

            if counter >= patience:
                print(f'Early stopping after {e+1} epochs')
                self.num_epochs = e+1
                break

            
            print(f'Epoch {e+1} \t\t Training Loss: {train_loss / len(train_dataloader)}')
            print(f'Epoch {e+1} \t\t Validation Loss: {valid_loss / len(val_dataloader)}') 
            # print(f'Epoch {e+1} \t\t Negative Penalty: {negative_penalty}') 
            # print(self.transf.weight)

        return train_loss/len(train_dataloader), valid_loss/len(val_dataloader)

    def freeze_abund_weights(self):
        for param in self.conv2.parameters():
            param.requires_grad = False

        for param in self.activate3.parameters():
            param.requires_grad = False

        for param in self.linear2.parameters():
            param.requires_grad = False

        for param in self.activate4.parameters():
            param.requires_grad = False

        return
    
    def unfreeze_act_weights(self):
        for param in self.conv1.parameters():
            param.requires_grad = True

        for param in self.activate1.parameters():
            param.requires_grad = True

        for param in self.linear1.parameters():
            param.requires_grad = True

        for param in self.activate2.parameters():
            param.requires_grad = True

        return

    def freeze_act_weights(self):
        for param in self.conv1.parameters():
            param.requires_grad = False

        for param in self.activate1.parameters():
            param.requires_grad = False

        for param in self.linear1.parameters():
            param.requires_grad = False

        for param in self.activate2.parameters():
            param.requires_grad = False

        return
    

    def train_model_act(self,train_dataloader,val_dataloader,epochs,device,learning_rate,outfile,neg_pen, weight_pen, act_pen):
        """
        Trains the model for a given number of epochs, stopping early if validation loss has not improved for five epochs.

        @param train_dataloader: Training data
        @param val_dataloader: Validation data
        @param epochs: Total number of epochs
        @param device: CPU or GPU depending on what's available
        @param learning_rate: Learning rate
        @param outfile: Name of saved model
        @return: The final loss
        """

        # Training parameters
        self.num_epochs = epochs
        self.outfile = outfile

        # Mean absolute error (average of sum of absolute differences)
        loss_function = torch.nn.L1Loss()

        # ADD THIS SO ONLY CERTAIN PARAMS ARE UPDATED
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()),lr=learning_rate)

        self.train_losses = []
        self.valid_losses = []

        # Initialize variables for early stopping
        best_metric = float('inf') 
        patience = 5  # Number of epochs with no improvement to wait before early stopping
        counter = 0  # Counter to keep track of epochs with no improvement

        # Initalize loss weight values
        activity_weight = act_pen
        abundance_weight = 2 - activity_weight

        for e in range(self.num_epochs):
            train_loss = 0.0
            
            # Set the model in training mode
            self.train()
            
            # Loop through training data, calculate loss and update weights
            for X,y in train_dataloader:
                X.to(device)
                y.to(device)
                
                X = X.to(torch.float32)
                
                # Caclulate the predicted values
                y_pred = self(X)

                k1, abund = self.get_ks()
                negative_penalty = torch.sum(torch.nn.ReLU()(-k1))
                weight_penalty = torch.sum(torch.nn.ReLU()(-self.linear1.weight))

                # print(y_pred.shape) # torch.Size([10, 2, 1])
                y = y.reshape([y.shape[0], y.shape[1], y.shape[2]])
                # print(y.shape) # torch.Size([10, 2, 1, 1])
                
                # Calculate the difference between the predicted values and the actual values
                # ONLY TRAIN ON ACTIVITY
                loss = loss_function(y_pred[:,1], y[:,1]) + neg_pen * negative_penalty + weight_pen * weight_penalty                # loss = negative_penalty + weight_penalty

                optimizer.zero_grad()
                loss.backward() 
                optimizer.step()

                # Calculate loss
                train_loss += loss.item()

            self.train_losses.append(train_loss/len(train_dataloader))

            valid_loss = 0.0

            # Set the model to eval mode
            self.eval()    

            # Loop through the validation data and calculate loss (no weight updates)
            for X, y in val_dataloader:    
                y_pred = self(X)
                k1, abund = self.get_ks()
                negative_penalty = torch.sum(torch.nn.ReLU()(-k1))
                weight_penalty = torch.sum(torch.nn.ReLU()(-self.linear1.weight))

                y = y.reshape([y.shape[0], y.shape[1], y.shape[2]])

                # ONLY TRAIN ON ACTIVITY
                loss = loss_function(y_pred[:,1], y[:,1]) + neg_pen * negative_penalty + weight_pen * weight_penalty
                # loss = negative_penalty + weight_penalty
                valid_loss += loss.item()

            self.valid_losses.append(valid_loss/len(val_dataloader))

                        # Check for early stopping
            if valid_loss < best_metric:
                best_metric = valid_loss
                counter = 0
                torch.save(self.state_dict(), f'{outfile}.pth')
            else:
                counter += 1

            if counter >= patience:
                print(f'Early stopping after {e+1} epochs')
                self.num_epochs = e+1
                break

            print(f'Epoch {e+1} \t\t Training Loss: {train_loss / len(train_dataloader)}')
            print(f'Epoch {e+1} \t\t Validation Loss: {valid_loss / len(val_dataloader)}') 
            # print(f'Epoch {e+1} \t\t Negative Penalty: {negative_penalty}') 
    
    def get_actual_and_predicted(self,dataloader,device,filepath):
        """
        Creates a CSV file with the actual and predicted values for each sequence.
        @param dataloader: Data to give to model
        @param device: Either CPU or GPU
        @param filepath: Where to save the results
        """
        print(self.outfile)
        all_data = pd.DataFrame([])
        print(self.transf)
        best_model = self.__class__(self.input_shape,self.transformation,self.kernel_size,self.outchannel,self.relu,self.positive, self.hill_val)
        best_model.load_state_dict(torch.load(f'{self.outfile}.pth'))
        best_model.eval()

        for X,y in dataloader:
            X.to(device)
            y.to(device)
            
            X = X.to(torch.float32) # Added cause was getting error cause diff type than internal weights
            
            sequences = pd.DataFrame(one_hot_to_sequence(X))
            # Caclulate the predicted values
            y_pred = best_model(X)
            abund_pred = pd.DataFrame(y_pred[:,0].detach().numpy().flatten())
            activity_pred = pd.DataFrame(y_pred[:,1].detach().numpy().flatten())
            abund_actual = pd.DataFrame(y[:,0].detach().numpy().flatten())
            activity_actual = pd.DataFrame(y[:,1].detach().numpy().flatten())

            data = pd.concat([abund_pred,abund_actual,activity_pred,activity_actual,sequences],axis=1)
            all_data = pd.concat([all_data, data])
        all_data.columns = ["abund_pred","abund_actual","activity_pred","activity_actual","aa_seq"]
        all_data.to_csv(filepath,index=False)

    def get_losses(self,outfile):
        """
        Plots the losses of each epoch and saves plot and values.
        @param outfile: Where to save the final figure and values. 
        """
        epochs = range(self.num_epochs)
        plt.plot(epochs,self.valid_losses,label="Test Loss")
        plt.plot(epochs,self.train_losses, label="Train Loss")
        plt.legend(loc="upper left")
        plt.xlabel("Epochs")
        plt.ylabel("Losses")
        # plt.show()
        plt.savefig(f'{outfile}.png')

        df = pd.DataFrame([range(self.num_epochs),self.valid_losses, self.train_losses])
        df = df.transpose()
        df.columns=['epochs','valid_loss','train_loss']

        df.to_csv(f'{outfile}_losses.csv')

class ADModel_two_state_abund(ADModel_two_state):
    """
    Same as ADModel except for instead of predicting closed, this predicts abundance directly 
    Uses a equlibrium model and a convolutional layer. 
    To use a single linear layer instead of a convolutional layer, set kernel_size to 40.
    """
    def __init__(self,input_shape,transformation,kernel_size=10,outchannel=1,relu = True,positive=False, hill_val=1, abund_k=15, seed=25):
        """
        Initalizes the ADModel object.
        @param input_shape: The size of the input. 2D input is expected, so size should be a tuple of size 2 with (rows,columns)
        @param transformation: Function to go from bound to activity. One of the following options: 'Linear', 'Hill' or 'Exponential.' 
        @param kernel_size: Size of the kernel i.e. how many adjacent amino acids should be considered at a time. 40 turns kernel into a single linear layer.
        @param outchannel: How many different kernels to use. Outchannel of 1 is most interpretable but slightly less good.
        @param relu: True indicates to use the ReLU activation function, while False indicates using the Parametric ReLU function.
        """
        super(ADModel_two_state_abund, self).__init__(input_shape, transformation, kernel_size, outchannel, relu, positive, hill_val) # initalizing parent class (torch.nn.Module)
        torch.manual_seed(seed)

        self.input_shape = input_shape
        self.relu = relu
        self.positive = positive
        self.abund_k = abund_k
        self.transformation = transformation
        self.kernel_size = kernel_size
        self.outchannel = outchannel
        self.hill_val = hill_val

        # Layers to predict inactive
        self.conv2 = torch.nn.Conv2d(in_channels=1, out_channels=self.outchannel, kernel_size=(abund_k,20))
        torch.nn.init.kaiming_normal_(self.conv2.weight, mode='fan_in', nonlinearity='leaky_relu') 
        torch.nn.init.constant_(self.conv2.bias, 0) 

        conv_out_height =self.input_shape[1] - abund_k + 1
        conv_out_width = self.input_shape[2] - 20 + 1
        linear_input_size = conv_out_height * conv_out_width * self.outchannel
        self.linear2 = torch.nn.Linear(in_features=linear_input_size, out_features=1)
        torch.nn.init.kaiming_normal_(self.linear2.weight, mode='fan_in', nonlinearity='leaky_relu') 
        torch.nn.init.constant_(self.linear2.bias, 0) 

        # Parameters for convolutional layers
        kernel_size = (kernel_size,20) # (sequence position, amino acid)
        conv_out_height =self.input_shape[1] - kernel_size[0] + 1
        conv_out_width = self.input_shape[2] - kernel_size[1] + 1
        linear_input_size = conv_out_height * conv_out_width * self.outchannel
        
        # Layers to predict K
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=self.outchannel, kernel_size=kernel_size)
        torch.nn.init.kaiming_normal_(self.conv1.weight, mode='fan_in', nonlinearity='leaky_relu') 
        torch.nn.init.constant_(self.conv1.bias, 0) 
        
        self.linear1 = torch.nn.Linear(in_features=linear_input_size, out_features=1)
        torch.nn.init.kaiming_normal_(self.linear1.weight, mode='fan_in', nonlinearity='leaky_relu') 
        torch.nn.init.constant_(self.linear1.bias, 0) 


        # Shared activation function
        if self.relu:    
            self.activate1 = torch.nn.ReLU()
            self.activate3 = torch.nn.ReLU()
        else:
            self.activate1 = torch.nn.PReLU(num_parameters=1, init=0.25)
            self.activate2 = torch.nn.PReLU(num_parameters=1, init=0.25)
            self.activate3 = torch.nn.PReLU(num_parameters=1, init=0.25)
            self.activate4 = torch.nn.PReLU(num_parameters=1, init=0.25)

        self.sigmoid = torch.nn.Sigmoid()
        
        # Layer to go from bound to activity
        if transformation == "Linear":
            self.transf = torch.nn.Linear(1, 1)
            self.transf.reset_parameters() 
        elif transformation == "Hill":
            self.transf = HillActivation(hill_val)
            self.transf.reset_parameters()
        else:
            print("Not a valid transformation, defaulting to Linear")
            self.transf = torch.nn.Linear(1, 1)

    def forward(self,x): 
        """
        Applies the model to the input tensor to get output predictions. 
        @param x: The input tensor
        @return: A tensor containing abundance and activity predictions. 
        """
        x = x.float()

        # Estimating K
        K = self.conv1(x)
        K = self.activate1(K)
        K = K.view(K.size(0),K.size(1)*K.size(2)*K.size(3)) # Linear layers only operate on 2d tensors
        K = self.linear1(K)
        if not self.relu: 
            K = self.activate2(K)

        # Estimating inactive
        abund = self.conv2(x)
        abund = self.activate3(abund)
        abund = abund.view(abund.size(0),abund.size(1)*abund.size(2)*abund.size(3)) # Linear layers only operate on 2d tensors
        abund = self.linear2(abund)
        if not self.relu:
            abund = self.activate4(abund)

        # Saving Ks for future access
        self.K = K
        self.abund = abund
        
        bound = torch.div(abund, torch.add(1, torch.div(1,K)))
        activity = self.transf(bound)
        # activity = k2
        return torch.stack((abund, activity)).transpose(0,1)
    
    def get_ks(self):
        """
        Run directly after calling forward on tensor of interest. 
        @return: The K1, K2, and closed values for the input tensor x. 
        """
        return self.K, self.abund

    def train_model_abund(self,train_dataloader,val_dataloader,epochs,device,learning_rate,outfile, neg_pen, weight_pen, act_pen):
        """
        Trains the model for a given number of epochs, stopping early if validation loss has not improved for five epochs.

        @param train_dataloader: Training data
        @param val_dataloader: Validation data
        @param epochs: Total number of epochs
        @param device: CPU or GPU depending on what's available
        @param learning_rate: Learning rate
        @param outfile: Name of saved model
        @return: The final loss
        """

        # Training parameters
        self.num_epochs = epochs
        self.outfile = outfile

        # Mean absolute error (average of sum of absolute differences)
        loss_function = torch.nn.L1Loss()
        # loss_function = torch.nn.MSELoss()

        optimizer = torch.optim.Adam(params=self.parameters(),lr=learning_rate)

        self.train_losses = []
        self.valid_losses = []

        # Initialize variables for early stopping
        best_metric = float('inf') 
        patience = 5  # Number of epochs with no improvement to wait before early stopping
        counter = 0  # Counter to keep track of epochs with no improvement

        # Initalize loss weight values
        activity_weight = act_pen
        abundance_weight = 2 - activity_weight

        for e in range(self.num_epochs):
            train_loss = 0.0
            
            # Set the model in training mode
            self.train()
            
            # Loop through training data, calculate loss and update weights
            for X,y in train_dataloader:
                X.to(device)
                y.to(device)
                
                X = X.to(torch.float32)
                
                # Caclulate the predicted values
                y_pred = self(X)

                k1, abund = self.get_ks()
                negative_penalty = torch.sum(torch.nn.ReLU()(-abund))
                weight_penalty = torch.sum(torch.nn.ReLU()(-self.linear2.weight)) #+ torch.nn.ReLU()(-self.transf.weight))

                y = y.reshape([y.shape[0], y.shape[1], y.shape[2]])

                # ONLY TRAIN ON ABUND
                loss = loss_function(y_pred[:,0], y[:,0]) + neg_pen * negative_penalty + weight_pen * weight_penalty 
                # loss = negative_penalty + weight_penalty

                optimizer.zero_grad()
                loss.backward() 
                optimizer.step()

                # Calculate loss
                train_loss += loss.item()

            self.train_losses.append(train_loss/len(train_dataloader))

            valid_loss = 0.0

            # Set the model to eval mode
            self.eval()    

            # Loop through the validation data and calculate loss (no weight updates)
            for X, y in val_dataloader:    
                y_pred = self(X)

                k1, abund = self.get_ks()
                negative_penalty = torch.sum(torch.nn.ReLU()(-abund))
                weight_penalty = torch.sum(torch.nn.ReLU()(-self.linear2.weight)) #+ torch.nn.ReLU()(-self.transf.weight))

                y = y.reshape([y.shape[0], y.shape[1], y.shape[2]])

                # ONLY TRAINING ON ABUNDANCE LOSS
                loss = loss_function(y_pred[:,0], y[:,0]) + neg_pen * negative_penalty + weight_pen * weight_penalty 
                valid_loss += loss.item()

            self.valid_losses.append(valid_loss/len(val_dataloader))

            # Check for early stopping
            if valid_loss < best_metric:
                best_metric = valid_loss
                counter = 0
                torch.save(self.state_dict(), f'{outfile}.pth')
            else:
                counter += 1

            if counter >= patience:
                print(f'Early stopping after {e+1} epochs')
                self.num_epochs = e+1
                break

            
            print(f'Epoch {e+1} \t\t Training Loss: {train_loss / len(train_dataloader)}')
            print(f'Epoch {e+1} \t\t Validation Loss: {valid_loss / len(val_dataloader)}') 
            # print(f'Epoch {e+1} \t\t Negative Penalty: {negative_penalty}') 
            # print(self.transf.weight)

        return train_loss/len(train_dataloader), valid_loss/len(val_dataloader)

    def freeze_abund_weights(self):
        for param in self.conv2.parameters():
            param.requires_grad = False

        for param in self.activate3.parameters():
            param.requires_grad = False

        for param in self.linear2.parameters():
            param.requires_grad = False

        for param in self.activate4.parameters():
            param.requires_grad = False

        return
    
    def unfreeze_act_weights(self):
        for param in self.conv1.parameters():
            param.requires_grad = True

        for param in self.activate1.parameters():
            param.requires_grad = True

        for param in self.linear1.parameters():
            param.requires_grad = True

        for param in self.activate2.parameters():
            param.requires_grad = True

        return

    def freeze_act_weights(self):
        for param in self.conv1.parameters():
            param.requires_grad = False

        for param in self.activate1.parameters():
            param.requires_grad = False

        for param in self.linear1.parameters():
            param.requires_grad = False

        for param in self.activate2.parameters():
            param.requires_grad = False

        return
    

    def train_model_act(self,train_dataloader,val_dataloader,epochs,device,learning_rate,outfile,neg_pen, weight_pen, act_pen):
        """
        Trains the model for a given number of epochs, stopping early if validation loss has not improved for five epochs.

        @param train_dataloader: Training data
        @param val_dataloader: Validation data
        @param epochs: Total number of epochs
        @param device: CPU or GPU depending on what's available
        @param learning_rate: Learning rate
        @param outfile: Name of saved model
        @return: The final loss
        """

        # Training parameters
        self.num_epochs = epochs
        self.outfile = outfile

        # Mean absolute error (average of sum of absolute differences)
        loss_function = torch.nn.L1Loss()

        # ADD THIS SO ONLY CERTAIN PARAMS ARE UPDATED
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()),lr=learning_rate)

        self.train_losses = []
        self.valid_losses = []

        # Initialize variables for early stopping
        best_metric = float('inf') 
        patience = 5  # Number of epochs with no improvement to wait before early stopping
        counter = 0  # Counter to keep track of epochs with no improvement

        # Initalize loss weight values
        activity_weight = act_pen
        abundance_weight = 2 - activity_weight

        for e in range(self.num_epochs):
            train_loss = 0.0
            
            # Set the model in training mode
            self.train()
            
            # Loop through training data, calculate loss and update weights
            for X,y in train_dataloader:
                X.to(device)
                y.to(device)
                
                X = X.to(torch.float32)
                
                # Caclulate the predicted values
                y_pred = self(X)

                k1, abund = self.get_ks()
                negative_penalty = torch.sum(torch.nn.ReLU()(-k1))
                weight_penalty = torch.sum(torch.nn.ReLU()(-self.linear1.weight))

                # print(y_pred.shape) # torch.Size([10, 2, 1])
                y = y.reshape([y.shape[0], y.shape[1], y.shape[2]])
                # print(y.shape) # torch.Size([10, 2, 1, 1])
                
                # Calculate the difference between the predicted values and the actual values
                # ONLY TRAIN ON ACTIVITY
                loss = loss_function(y_pred[:,1], y[:,1]) + neg_pen * negative_penalty + weight_pen * weight_penalty                # loss = negative_penalty + weight_penalty

                optimizer.zero_grad()
                loss.backward() 
                optimizer.step()

                # Calculate loss
                train_loss += loss.item()

            self.train_losses.append(train_loss/len(train_dataloader))

            valid_loss = 0.0

            # Set the model to eval mode
            self.eval()    

            # Loop through the validation data and calculate loss (no weight updates)
            for X, y in val_dataloader:    
                y_pred = self(X)
                k1, abund = self.get_ks()
                negative_penalty = torch.sum(torch.nn.ReLU()(-k1))
                weight_penalty = torch.sum(torch.nn.ReLU()(-self.linear1.weight))

                y = y.reshape([y.shape[0], y.shape[1], y.shape[2]])

                # ONLY TRAIN ON ACTIVITY
                loss = loss_function(y_pred[:,1], y[:,1]) + neg_pen * negative_penalty + weight_pen * weight_penalty
                # loss = negative_penalty + weight_penalty
                valid_loss += loss.item()

            self.valid_losses.append(valid_loss/len(val_dataloader))

                        # Check for early stopping
            if valid_loss < best_metric:
                best_metric = valid_loss
                counter = 0
                torch.save(self.state_dict(), f'{outfile}.pth')
            else:
                counter += 1

            if counter >= patience:
                print(f'Early stopping after {e+1} epochs')
                self.num_epochs = e+1
                break

            print(f'Epoch {e+1} \t\t Training Loss: {train_loss / len(train_dataloader)}')
            print(f'Epoch {e+1} \t\t Validation Loss: {valid_loss / len(val_dataloader)}') 
            # print(f'Epoch {e+1} \t\t Negative Penalty: {negative_penalty}') 
    
    def get_actual_and_predicted(self,dataloader,device,filepath):
        """
        Creates a CSV file with the actual and predicted values for each sequence.
        @param dataloader: Data to give to model
        @param device: Either CPU or GPU
        @param filepath: Where to save the results
        """
        print(self.outfile)
        all_data = pd.DataFrame([])
        print(self.transf)
        best_model = self.__class__(self.input_shape,self.transformation,self.kernel_size,self.outchannel,self.relu,self.positive, self.hill_val, self.abund_k)
        best_model.load_state_dict(torch.load(f'{self.outfile}.pth'))
        best_model.eval()

        for X,y in dataloader:
            X.to(device)
            y.to(device)
            
            X = X.to(torch.float32) # Added cause was getting error cause diff type than internal weights
            
            sequences = pd.DataFrame(one_hot_to_sequence(X))
            # Caclulate the predicted values
            y_pred = best_model(X)
            abund_pred = pd.DataFrame(y_pred[:,0].detach().numpy().flatten())
            activity_pred = pd.DataFrame(y_pred[:,1].detach().numpy().flatten())
            abund_actual = pd.DataFrame(y[:,0].detach().numpy().flatten())
            activity_actual = pd.DataFrame(y[:,1].detach().numpy().flatten())

            data = pd.concat([abund_pred,abund_actual,activity_pred,activity_actual,sequences],axis=1)
            all_data = pd.concat([all_data, data])
        all_data.columns = ["abund_pred","abund_actual","activity_pred","activity_actual","aa_seq"]
        all_data.to_csv(filepath,index=False)


def get_params(args):
    """
    Reads in command line arguments and stores them as variables.
    @param args: Command line argument object
    @return: the file, scale, batch size, activity function, epochs, out file name, number of kernels, width of kernel, learning rate, and whether to use relu
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
    positive = args.positive
    
    if args.negative_penalty:
        neg_pen = args.negative_penalty
    else:
        neg_pen = 0 

    if args.weight_penalty:
        weight_pen = args.weight_penalty
    else:
        weight_pen = 0 

    if args.activity_penalty:
        act_pen = args.activity_penalty
    else:
        act_pen = 1 

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

    return file, scale, batch_size, activity_fun, epochs, outfile, outchannel, kernel_size, learning_rate, relu, test, positive, val_file, intelligent_split, neg_pen, weight_pen, act_pen, hill_val, abund_kernel_value, seed

# Run the model in the normal way
def fit(args): 
    """
    Loads the data, trains the model, gets the actual and predicted values, saves the model, and plots the losses.
    @param args: command line arguments
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
    test = params[10] 
    positive = params[11] 
    val_file = params[12]
    intelligent_split = params[13]
    neg_pen = params[14]
    weight_pen = params[15] 
    act_pen = params[16]
    hill_val = params[17]
    abund_kernel_value = params[18]
    seed = params[19]

    data_reader = DataReader()
    
    # Split data into train and testing sets and create data loaders for each group
    split_data = SplitData(data_reader,encoding_type="2D")
    
    print(seed)
    if intelligent_split: 
        split_data.read_split_data(file, val_file, scaler=scale)

    else: 
        split_data.read_data(scale=scale)
        split_data.split_data(ratio=False)
    # train_data_loader, val_data_loader, test_data_loader, size = split_data.load_data(batch_size)
    train_data_loader, val_data_loader, size = split_data.load_data(batch_size)
    
    # Create the model
    if args.model == "three_state":
        print(f"Training three state model with filter size of {kernel_size}")
        model = ADModel_three_state(size, activity_fun, kernel_size, outchannel, relu, positive, hill_val, seed=seed)
    elif args.model == "three_state_abund":
        print(f"Training three state model with filter size of {kernel_size} and abundance filter size of {abund_kernel_value}")
        model = ADModel_three_state_abund(size, activity_fun, kernel_size, outchannel, relu, positive, hill_val, abund_k=abund_kernel_value, seed=seed)
    elif args.model == "two_state":
        print(f"Training two state model with filter size of {kernel_size}")
        model = ADModel_two_state(size, activity_fun, kernel_size, outchannel, relu, positive, hill_val, seed=seed)
    elif args.model == "two_state_abund":
        print(f"Training two state model with filter size of {kernel_size} and abundance filter size of {abund_kernel_value}")
        model = ADModel_two_state_abund(size, activity_fun, kernel_size, outchannel, relu, positive, hill_val, abund_k=abund_kernel_value, seed=seed)
    elif args.model == "simple_act":
        print(f"Training simple activity model with filter size of {kernel_size}")
        model = ADModel_act(size,kernel_size, seed=seed)
    elif args.model == "simple_abund":
        print(f"Training simple abundance model with filter size of {kernel_size}")
        model = ADModel_abund(size,kernel_size, seed=seed)
    else: 
        print(f"Defaulting to three_state model with filter size of {kernel_size}")
        model = ADModel_three_state(size,activity_fun,kernel_size,outchannel,relu,positive,hill_val, seed=seed)

    # Use GPUs if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)
    model.to(device)

    if (args.model == "simple_act") or (args.model == "simple_abund"):
        print("Training simple module")
        model.train_model(train_data_loader,val_data_loader,epochs,device,learning_rate,outfile, weight_pen)
    else:

        # Train the model using the loaded data, automatically saves the best mile as outfile
        print("Training abundance module")
        model.freeze_act_weights()
        model.train_model_abund(train_data_loader,val_data_loader,epochs,device,learning_rate,outfile, 0, 1, act_pen)

        model.eval()
        # model.get_actual_and_predicted(val_data_loader, device, filepath=f"{outfile}_vals.csv", save_file=False)

        print("Training activity module")
        model.train()
        model.freeze_abund_weights()
        model.unfreeze_act_weights()
        model.train_model_act(train_data_loader,val_data_loader,epochs,device,learning_rate,outfile, neg_pen, weight_pen, act_pen)
        model.freeze_act_weights()

    model.eval()
    # Get final model predictions
    model.get_actual_and_predicted(val_data_loader, device,filepath=f"{outfile}_vals.csv")
    model.get_actual_and_predicted(train_data_loader, device, filepath=f"{outfile}_train.csv")
    if test:
        model.get_actual_and_predicted(test_data_loader, device,filepath=f"{outfile}_test.csv")

    # Plot test and train losses
    model.get_losses(outfile)

def main():
    # Initialize parser
    parser = argparse.ArgumentParser()

    # Adding arguments related to model training
    parser.add_argument("-f", "--file", help = "Input file", type=str)
    parser.add_argument("-v", "--val_file", help = "Input file to validation sequences", type=str)
    parser.add_argument("-b", "--batch", help = "Batch size",type=int)
    parser.add_argument("-l", "--learning_rate", help = "Learning rate",type=float)
    parser.add_argument("-e", "--epochs", help = "Number of epochs", type=int)
    parser.add_argument("-o", "--outfile", help = "File to write predicted data", type=str)
    parser.add_argument("-t","--test",action='store_true',help="Whether to evaluate on test data as well")
    parser.add_argument("-i","--intelligent_split",action='store_true',help="Whether to use separate file of validation data (must be provided with the -v argument)")
    parser.add_argument("-np","--negative_penalty",help="How much to penalize negative K values",type=float)
    parser.add_argument("-wp","--weight_penalty",help="How much to penalize negative weight values", type=float)
    parser.add_argument("-ap","--activity_penalty",help="How much to weight the activity loss (abundance loss gets weight of 1-[this value])", type=float)

    # Adding arguments related to the actual model
    parser.add_argument("-s", "--scale",help = "Method for scaling the activity and abundance",type=str)
    parser.add_argument("-a", "--activity", help = "Function used to calcualte activity", type=str)
    parser.add_argument("-c", "--outchannels", help = "File to write predicted data", type=int)
    parser.add_argument("-k","--kernel_size",help="Width of kernel",type=int)
    parser.add_argument("-r","--relu",action='store_true',help="Use the ReLU activation function instead of the parametric relu")
    parser.add_argument("-p","--positive",action='store_true',help="Only use positive weights")
    parser.add_argument("-m","--model",help="Which model to run. Options are 'closed', 'abundance', or 'two-state'",type=str)
    parser.add_argument("-hv", "--hill_value", help="What n value to use in the hill function", type=int)
    parser.add_argument("-ak", "--abund_kernel_value", help="What kernel size to use to predict the activity", type=int)

    # seed
    parser.add_argument("-seed", "--seed", help="Initalization seed to use", type=int)


    # Read arguments from command line
    args = parser.parse_args()
    fit(args)
    
if __name__ == "__main__":
    main()