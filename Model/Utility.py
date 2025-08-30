"""
Modified Date: 29 August 2025
Author: Claire LeBlanc

Usage: python Final_Model.py [optional args]
"""

import torch
torch.manual_seed(25)


def one_hot_to_sequence(one_hot_tensor):
    """
    Convert a one-hot encoded tensor of protein sequences to the original sequences.

    Parameters
    ----------
        one_hot_tensor : torch.Tensor
            A tensor of shape (num_sequences, sequence_length, num_amino_acids)
            representing one-hot encoded protein sequences.

    Returns
    -------
        list of stings
            A list of the original protein sequences.
    """
    # Check if the tensor is on GPU and move it to CPU if necessary
    if one_hot_tensor.is_cuda:
        one_hot_tensor = one_hot_tensor.cpu()

    amino_acids = "RHKDESTNQCGPAVILMFYW"

    # Find the indices of the maximum values (i.e the ones) along the last dimension
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

    Attributes
    ----------
    beta : torch.Parameter
        The beta parameter of the hill function, representing the max expression value

    K : torch.Parameter
        The K parameter of the hill function, representing the midpoint of the hill curve

    n : int
        The hill coefficient, representing the cooperativity
    """
    
    def __init__(self, n=1):
        """
        Initalizes the class. Defines traininable parameters beta and K.

        Parameters
        ----------
        n : int
            The hill coefficient, defines cooperativity in biological interaction

        """
        super(HillActivation, self).__init__()
        torch.manual_seed(25)

        # Define the three parameters we want the neural network to learn
        self.beta = torch.nn.Parameter(torch.Tensor([4.0])) #Beta is max expression value
        self.K = torch.nn.Parameter(torch.Tensor([2.0])) #K is midpoint of hill curve, assuming no basal expression, half of beta
        self.n = float(n)  # Need this to be only integer values i.e. 1,2,4
        

    def forward(self, x):
        """
        Applies hill function to tensor x.

        Parameters
        ----------
        x : torch.Tensor
            The tensor to apply the hill function to

        Returns
        -------
        torch.Tensor
            New tensor with hill function applied
        """

        # Formula for the hill function: (beta * x^n) / (K^n + x^n)
        xn = torch.pow(x, self.n) 
        kn = torch.pow(self.K, self.n)

        numerator = torch.mul(self.beta, xn)
        denom = torch.add(kn, xn)

        output = torch.div(numerator, denom)

        return output
    
    def reset_parameters(self):
        """
        Resets trainable parameters, beta and K.
        """
        self.beta.data = torch.Tensor([4.0]) #Beta is max expression value, starting this with max activity (i.e. fluorescence) in dataset
        self.K.data = torch.Tensor([2.0]) #K is midpoint of hill curve, assuming no basal expression, half of beta

    def freeze_hill_params(self):
        """
        Freezes trainable parameters, beta and K.
        """   
        self.beta.requires_grad = False
        self.K.requires_grad = False
    
    def unfreeze_hill_params(self):
        """
        Unfreezes trainable parameters, beta and K.
        """   
        self.beta.requires_grad = True
        self.K.requires_grad = True