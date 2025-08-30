"""
Modified Date: 29 August 2025
Author: Claire LeBlanc
"""

import torch
import pandas as pd
import matplotlib.pyplot as plt
from Utility import one_hot_to_sequence
torch.manual_seed(25)

class ADModel_abund(torch.nn.Module):
    """
    Class for implementing the mCherry SimpleNN. 
    Code to create and train a NN that predicts mCherry signal 
    from one hot encoded amino acid sequence. 

    Attributes
    ----------
    input_shape : tuple
        The size of the one hot encoded amino acid sequence. For
        all of our purposes this should be (40, 20)

    kernel_size : int
        The size of the convolutional filter, can range from 1-40
    
    conv1 : torch.nn.Conv2d
        The convolutional layer
    
    linear1 : torch.nn.Linear
        The dense/fully connected layer
    
    activate : torch.nn.PReLU
        The parametric ReLU activation function

    train_losses : list
        Stores the trainning data loss for each epoch

    val_losses : list
        Stores the validation data loss for each epoch

    """
    def __init__(self,input_shape,kernel_size=10, seed=25):
        """
        Initalizes the ADModel object.

        Parameters
        ----------
        input_shape : tuple
            The size of the input. 2D input is expected, so size should be a tuple of size 2 with (rows,columns)

        kernel_size : int
            Size of the kernel i.e. how many adjacent amino acids should be considered at a time. 40 turns kernel into a single linear layer.

        seed : int
            Random seed used to initalize weights
        """
        super(ADModel_abund, self).__init__() # initalizing parent class (torch.nn.Module)
        torch.manual_seed(seed)
        
        self.input_shape = input_shape
        self.kernel_size = kernel_size

        # Parameters for convolutional layers
        out_channels = 1
        kernel_size = (kernel_size,20) # (sequence position, amino acid)

        # Calculating how many linear input nodes are needed
        conv_out_height = self.input_shape[1] - kernel_size[0] + 1
        conv_out_width = self.input_shape[2] - kernel_size[1] + 1
        linear_input_size = conv_out_height * conv_out_width * out_channels
        
        # Layers to predict mCherry
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=kernel_size)
        
        # Initalizing the weights in a consistent way to try to decrease variability
        torch.nn.init.kaiming_normal_(self.conv1.weight, mode='fan_in', nonlinearity='leaky_relu') 
        torch.nn.init.constant_(self.conv1.bias, 0)

        self.linear1 = torch.nn.Linear(in_features=linear_input_size, out_features=1)

        # Initalizing the weights in a consistent way to try to decrease variability
        torch.nn.init.kaiming_normal_(self.linear1.weight, mode='fan_in', nonlinearity='leaky_relu') 
        torch.nn.init.constant_(self.linear1.bias, 0)    # All biases set to 0.5

        # Activation function 
        self.activate = torch.nn.PReLU(num_parameters=1, init=0.25)


    def forward(self,x): 
        """
        Applies the model to the input tensor to get output predictions. 

        Parameters
        ----------
        x : torch.Tensor
            The input tensor
        
        Returns
        --------
        torch.Tensor
            A tensor containing abundance predictions. 
        """
        x = x.float()

        # Predicting abundance
        abund = self.conv1(x) # convolutional filter
        abund = self.activate(abund) # activation function
        abund = abund.view(abund.size(0),abund.size(1)*abund.size(2)*abund.size(3)) # Linear layers only operate on 2d tensors
        abund = self.linear1(abund) # linear layer
        abund = self.activate(abund) # activation function

        return abund
    
    
    def train_model(self,train_dataloader,val_dataloader,epochs,device,learning_rate,outfile,weight_pen=0):
        """
        Trains the model for a given number of epochs, stopping early if validation loss has not improved for five epochs.

        Parameters
        ----------
        train_dataloader : FastTensorDataLoader
            Training data

        val_dataloader : FastTensorDataLoader
            Validation data

        epochs : int
            Total number of epochs

        device : str
            CPU or GPU depending on what's available

        learning_rate : float
            Learning rate

        outfile : str
            Name of saved model
        
        weight_pen : float
            How much to penalize negative linear weights 
            (defaults to 0, no penalty)
        
        Returns
        -------
        float 
            The final loss
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
                
                # Run model on input tensor
                y_pred = self(X)

                # Adds all the negative linear weights
                weight_penalty = torch.sum(torch.nn.ReLU()(-self.linear1.weight))
                
                y = y.reshape([y.shape[0], y.shape[1], y.shape[2]])
                
                # Compare predicted value with measured abundance value
                loss = loss_function(y_pred, y[:,0]) + weight_pen * weight_penalty

                # Update weights
                optimizer.zero_grad() # reset gradient
                loss.backward() # compute gradient of loss
                optimizer.step() # updates parameters based on gradient

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

             # Prints loss after each epoch
            print(f'Epoch {e+1} \t\t Training Loss: {train_loss / len(train_dataloader)}')
            print(f'Epoch {e+1} \t\t Validation Loss: {valid_loss / len(val_dataloader)}') 

        return train_loss/len(train_dataloader), valid_loss/len(val_dataloader)

    
    def get_actual_and_predicted(self,dataloader,device,filepath, save_file=True):
        """
        Creates a CSV file with the actual and predicted values for each sequence.

        Parameters
        ----------
        dataloader : FastTensorDataLoader
            Data to give to model

        device : str
            Either CPU or GPU
        
        filepath : str
            Where to save the results
        
        save_file : bool
            Whether or not to actually save file (added for debugging)
        """

        all_data = pd.DataFrame([])

        # Reloading model so that this method can be run without retrainning network
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
        
        Parameters
        ----------
        outfile : str
            Where to save the final figure and values. 
        """
        
        epochs = range(self.num_epochs)

        plt.plot(epochs,self.valid_losses,label="Test Loss")
        plt.plot(epochs,self.train_losses, label="Train Loss")
        plt.legend(loc="upper left")
        plt.xlabel("Epochs")
        plt.ylabel("Losses")
        plt.savefig(f'{outfile}.png')

        df = pd.DataFrame([range(self.num_epochs),self.valid_losses, self.train_losses])
        df = df.transpose()
        df.columns=['epochs','valid_loss','train_loss']

        df.to_csv(f'{outfile}_losses.csv')