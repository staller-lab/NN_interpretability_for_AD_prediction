"""
Modified Date: 29 August 2025
Author: Claire LeBlanc
"""

import torch
import pandas as pd
import matplotlib.pyplot as plt
from Utility import one_hot_to_sequence, HillActivation
torch.manual_seed(25)


class ADModel_three_state(torch.nn.Module):
    """
    Class for implementing the three state BiophysicalNN. 

    Code to create and train a NN that predicts TF abundance and gene 
    activation from one hot encoded amino acid sequence, and 
    incorporates a three state model of activation domain function.

    Note: In this class, the same kernel size is used to predict all 
    intermediate values (i.e. in all convolutional filters)

    Attributes
    ----------
    input_shape : tuple
        The size of the one hot encoded amino acid sequence. For
        all of our purposes this should be (40, 20)

    kernel_size : int
        The size of the convolutional filter, can range from 1-40
    
    relu : bool
        True indicates using the ReLU activation function, false uses
        the Parametric ReLU activation function

    outchannel : int
        The number of channels (i.e. kernels) for the convolutional filter

    hill_val : int
        The hill coefficient

    transformation : str
        Function for going from bound TF to gene expression
    
    is_linear : bool 
        Whether or not the transfrmation is linear

    conv1 : torch.nn.Conv2d
        The convolutional layer used to predict Kopen
    
    linear1 : torch.nn.Linear
        The dense layer used to predict Kopen

    conv2 : torch.nn.Conv2d
        The convolutional layer used to predict Kbound
    
    linear2 : torch.nn.Linear
        The dense layer used to predict Kbound

    conv3 : torch.nn.Conv2d
        The convolutional layer used to predict abundance
    
    linear3 : torch.nn.Linear
        The dense layer used to predict abundance
    
    activate : torch.nn.ReLU
        The ReLU activation function 
        
    activate (1-6) : torch.nn.PReLU
        Parametric ReLU activation functions

    train_losses : list
        Stores the trainning data loss for each epoch

    val_losses : list
        Stores the validation data loss for each epoch

    K1 : torch.Tensor
        The K1 (Kopen) values from the most recent prediction
    
    K2 : torch.Tensor
        The K2 (Kbound) values from the most recent prediction

    abundance : torch.Tensor
        The abundance values from the most recent prediction
    """

    def __init__(self,input_shape,transformation,kernel_size=10,outchannel=1,relu = True, hill_val=1, seed=25):
        """
        Initalizes the ADModel object.

        Parameters
        ----------
        input_shape : tuple
            The size of the input. 2D input is expected, so size should be a tuple of size 2 with (rows,columns)
        
        transformation: str
            Function to go from bound to activity. One of the following options: 'Linear', 'Hill' or 'Exponential.' 
        
        kernel_size: int
            Size of the kernel i.e. how many adjacent amino acids should be considered at a time. 40 turns kernel into a single linear layer.
        
        outchannel: int
            How many different kernels to use. Outchannel of 1 is most interpretable but slightly less good.
        
        relu: bool 
            True indicates to use the ReLU activation function, while False indicates using the Parametric ReLU function.
        
        hill val : int
            The hill coefficient

        seed : int
            Random seed used to initalize weights
        """
        
        super(ADModel_three_state, self).__init__() # initalizing parent class (torch.nn.Module)
        torch.manual_seed(seed)

        self.input_shape = input_shape
        self.relu = relu
        # self.positive = positive
        self.kernel_size = kernel_size
        self.outchannel = outchannel
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

        # self.sigmoid = torch.nn.Sigmoid()
        
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

        Parameters
        ----------
        x : torch.Tensor
            The input tensor
        
        Returns
        --------
        torch.Tensor
            A tensor containing abundance and activity predictions. 
        """
        x = x.float()

        # Estimating bound
        abund = self.conv3(x)
        abund = self.activate1(abund)
        abund = abund.view(abund.size(0),abund.size(1)*abund.size(2)*abund.size(3)) # Linear layers only operate on 2d tensors
        abund = self.linear3(abund)
        if not self.relu:
            abund = self.activate2(abund)

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

        # This is the equation derived relating abundance and the equilibrium constants to the amount of bound TF
        bound = torch.div(abund,torch.add(1,torch.mul(torch.div(1,k2),torch.add(1,torch.div(1,k1)))))

        # Apply transformation to go from bound to gene activation
        activity = self.transf(bound)

        return torch.stack((abund, activity)).transpose(0,1)
    
    def get_ks(self):
        """
        Run directly after calling forward on tensor of interest. 

        Returns
        -------
        K1 : torch.Tensor
            The K1 values for the most recent input tensor
        
        K2 : torch.Tensor
            The K2 values for the most recent input tensor
        
        abund : torch.Tensor
            The abundance values for the most recent input tensor
        """
        return self.K1, self.K2, self.abund
    
    def train_model_abund(self,train_dataloader,val_dataloader,epochs,device,learning_rate,outfile, neg_pen, weight_pen):
        """
        Trains the model for a given number of epochs, stopping early if validation loss has not improved for five epochs. 
        This specifically trains the abundance module, all other weights not involved in predicting abundance are frozen

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
        
        neg_pen : float
            How much to penalize negative equilibrium constant values
        
        Returns
        -------
        float 
            The final loss
        """

        # Freeze hill function weights
        if not self.islinear:
            # Do not learn the hill function right now
            self.transf.freeze_hill_params()

        # Training parameters
        self.num_epochs = epochs
        self.outfile = outfile

        # Mean absolute error (average of sum of absolute differences)
        loss_function = torch.nn.L1Loss()

        # only optimize weights that are not frozen
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()),lr=learning_rate)

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

                k1, k2, abund = self.get_ks()
                negative_penalty = torch.sum(torch.nn.ReLU()(-abund))
                weight_penalty = torch.sum(torch.nn.ReLU()(-self.linear3.weight)) 

                y = y.reshape([y.shape[0], y.shape[1], y.shape[2]])

                # Only train on abundance
                loss = loss_function(y_pred[:,0], y[:,0]) + neg_pen * negative_penalty + weight_pen * weight_penalty 

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

            # Prints loss after each epoch
            print(f'Epoch {e+1} \t\t Training Loss: {train_loss / len(train_dataloader)}')
            print(f'Epoch {e+1} \t\t Validation Loss: {valid_loss / len(val_dataloader)}') 

        return train_loss/len(train_dataloader), valid_loss/len(val_dataloader)

    def freeze_abund_weights(self):
        """
        Freezes the weights in the abundance layer
        """
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
        """
        Unfreezes the weights in the K1 and K2 layers
        """
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
        """
        Freezes the weights in the K1 and K2 layers
        """
                
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
    

    def train_model_act(self,train_dataloader,val_dataloader,epochs,device,learning_rate,outfile, neg_pen, weight_pen):
        """
        Trains the model for a given number of epochs, stopping early if validation loss has not improved for five epochs. 
        This specifically trains the K1 and K2 modules, all other weights not involved in predicting K1 and K2 are frozen

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
        
        neg_pen : float
            How much to penalize negative equilibrium constant values
        
        Returns
        -------
        float 
            The final loss
        """
       
        if not self.islinear:
            # Learn the hill function right now
            self.transf.unfreeze_hill_params()

        # Training parameters
        self.num_epochs = epochs
        self.outfile = outfile

        # Mean absolute error (average of sum of absolute differences)
        loss_function = torch.nn.L1Loss()

        # Only optimize the weights involved in K1 and K2
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()),lr=learning_rate)

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

                k1, k2, abund = self.get_ks()

                # Calulating the magnitude of negative equilibrium constants
                negative_penalty = torch.sum(torch.nn.ReLU()(-k1) + torch.nn.ReLU()(-k2))

                # Calculating the magnitude of negative weights
                weight_penalty = torch.sum(torch.nn.ReLU()(-self.linear1.weight)) + torch.sum(torch.nn.ReLU()(-self.linear2.weight))

                y = y.reshape([y.shape[0], y.shape[1], y.shape[2]])
                
                # Only trainning on activity values
                loss = loss_function(y_pred[:,1], y[:,1]) + neg_pen * negative_penalty + weight_pen * weight_penalty

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

                k1, k2, closed = self.get_ks()
                negative_penalty = torch.sum(torch.nn.ReLU()(-k1) + torch.nn.ReLU()(-k2))
                weight_penalty = torch.sum(torch.nn.ReLU()(-self.linear1.weight)) + torch.sum(torch.nn.ReLU()(-self.linear2.weight))

                y = y.reshape([y.shape[0], y.shape[1], y.shape[2]])

                # ONLY validate ON ACTIVITY
                loss = loss_function(y_pred[:,1], y[:,1]) + neg_pen * negative_penalty + weight_pen * weight_penalty
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
        best_model = self.__class__(self.input_shape,self.transformation,self.kernel_size,self.outchannel,self.relu,self.hill_val)
        best_model.load_state_dict(torch.load(f'{self.outfile}.pth'))
        best_model.eval()

        for X,y in dataloader:
            X.to(device)
            y.to(device)
            
            X = X.to(torch.float32) # Added cause was getting error from diff type than internal weights
            
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


class ADModel_three_state_abund(ADModel_three_state):
    """
    Class for implementing the three state BiophysicalNN. 

    Code to create and train a NN that predicts TF abundance and gene 
    activation from one hot encoded amino acid sequence, and 
    incorporates a three state model of activation domain function.

    Note: In this class, the abundance convolutional filter can be a
    different size than the kernels used to predict the equilibirum
    constants. 

    Attributes
    ----------
    input_shape : tuple
        The size of the one hot encoded amino acid sequence. For
        all of our purposes this should be (40, 20)

    kernel_size : int
        The size of the convolutional filter for K1 and K2, can range from 1-40

    abund_k : int
        The size of the convolutional filter for abundance, can range from 1-40
    
    relu : bool
        True indicates using the ReLU activation function, false uses
        the Parametric ReLU activation function

    outchannel : int
        The number of channels (i.e. kernels) for the convolutional filter

    hill_val : int
        The hill coefficient

    transformation : str
        Function for going from bound TF to gene expression
    
    is_linear : bool 
        Whether or not the transfrmation is linear

    conv1 : torch.nn.Conv2d
        The convolutional layer used to predict Kopen
    
    linear1 : torch.nn.Linear
        The dense layer used to predict Kopen

    conv2 : torch.nn.Conv2d
        The convolutional layer used to predict Kbound
    
    linear2 : torch.nn.Linear
        The dense layer used to predict Kbound

    conv3 : torch.nn.Conv2d
        The convolutional layer used to predict abundance
    
    linear3 : torch.nn.Linear
        The dense layer used to predict abundance
    
    activate : torch.nn.ReLU
        The ReLU activation function 
        
    activate (1-6) : torch.nn.PReLU
        Parametric ReLU activation functions

    train_losses : list
        Stores the trainning data loss for each epoch

    val_losses : list
        Stores the validation data loss for each epoch

    K1 : torch.Tensor
        The K1 (Kopen) values from the most recent prediction
    
    K2 : torch.Tensor
        The K2 (Kbound) values from the most recent prediction

    abundance : torch.Tensor
        The abundance values from the most recent prediction
    """

    def __init__(self,input_shape,transformation,kernel_size=10,outchannel=1,relu = True, hill_val=1, abund_k=15, seed=25):
        """
        Initalizes the ADModel object.

        Parameters
        ----------
        input_shape : tuple
            The size of the input. 2D input is expected, so size should be a tuple of size 2 with (rows,columns)
        
        transformation: str
            Function to go from bound to activity. One of the following options: 'Linear', 'Hill' or 'Exponential.' 
        
        kernel_size: int
            Size of the kernel for predicting K i.e. how many adjacent amino acids should be considered at a time. 40 turns kernel into a single linear layer.
        
        abund_k: int
            Size of the kernel for predicting abundance.
        
        outchannel: int
            How many different kernels to use. Outchannel of 1 is most interpretable but slightly less good.
        
        relu: bool 
            True indicates to use the ReLU activation function, while False indicates using the Parametric ReLU function.
        
        hill val : int
            The hill coefficient

        seed : int
            Random seed used to initalize weights
        """
        super(ADModel_three_state_abund, self).__init__(input_shape, transformation, kernel_size, outchannel, relu, hill_val) # initalizing parent class (ADModel_three_state)
        torch.manual_seed(seed)

        self.input_shape = input_shape
        self.relu = relu
        self.kernel_size = kernel_size
        self.outchannel = outchannel
        self.hill_val = hill_val
        self.transformation = transformation
        self.abund_k = abund_k # This controls the abundance filter size
        
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

        # Parameters for linear layers of K1 and K2
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
        print(f"Saving predictions to {self.outfile}")
        all_data = pd.DataFrame([])

        # Reloading model so that this method can be run without retrainning network
        best_model = self.__class__(self.input_shape,self.transformation,self.kernel_size,self.outchannel,self.relu,self.hill_val, self.abund_k)
        best_model.load_state_dict(torch.load(f'{self.outfile}.pth'))
        best_model.eval()

        for X,y in dataloader:
            X.to(device)
            y.to(device)
            
            X = X.to(torch.float32) # Added cause was getting error from diff type than internal weights
            
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