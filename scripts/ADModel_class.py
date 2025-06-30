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