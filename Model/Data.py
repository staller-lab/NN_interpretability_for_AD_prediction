"10/9/2023"

import random
import torch
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
import pandas as pd
import numpy as np
from joblib import dump, load

# This file contains three classes:
# - DataReader loads the file and extracts data into numpy dataframes
# - SplitData splits the data into test and train sets and loads the data into tensors
# - FastTensorDataLoader: Loads the tensors to prepare for training the model

# There are multiple options for data representation. Specifically: 
# - You can encode sequences as 1D or 2D matricies
# - You can load the predictions (abundance and activity) as a ratio or two separate values.

def one_hot_encode(seq):
    # Getting sequences
    matrix = np.zeros((len(seq),20))

    # Iterate over the amino acid sequence and set matrix entries to 1
    for position, amino_acid in enumerate(seq):
        # row_index = "ARNDCEQGHILKMFPSTWYV".index(amino_acid)
        row_index = "RHKDESTNQCGPAVILMFYW".index(amino_acid)
        matrix[position][row_index] = 1
    return np.array(matrix)

class DataReader:
    def __init__(
        self):

        self.seq = []
        self.data  =  []

    def load_file(self,file_path):
        self.data = pd.read_csv(file_path)

    # Encodes each sequence as a 800x1 matrix
    def encode_seq_1d(self):

        # Getting sequences
        X = self.data['aa_seq']

        #  Splitting sequences by character
        X = [ list(seq) for seq in X] 

        # Encoding the data -- column for each potential aa at each pos
        enc = OneHotEncoder(
                    handle_unknown='ignore',  
                    dtype = int)
        enc.fit(X)
        one_hot_df = np.array(enc.transform(X).toarray())

        self.seq = one_hot_df
        return one_hot_df
    
    # Encodes each sequence as a 40x20 matrix
    def encode_seq_2d(self):
         # Getting sequences
        X = self.data['aa_seq']
        one_hot = []
        for seq in X: 
            matrix = np.zeros((40,20))

            # Iterate over the amino acid sequence and set matrix entries to 1
            for position, amino_acid in enumerate(seq):
                # row_index = "ARNDCEQGHILKMFPSTWYV".index(amino_acid)
                row_index = "RHKDESTNQCGPAVILMFYW".index(amino_acid)
                matrix[position][row_index] = 1
            one_hot.append([matrix]) # Added extra dimension to get "channel" for convolution
        self.seq = np.array(one_hot)
        return np.array(one_hot)

    # Get abundance data
    def get_abundance(self):
        return self.data['abundance']

    def get_ratio(self):
        return self.data['ratio']
    
    # Get activity data
    def get_activity(self):
        return self.data['activity']


class SplitData:
    def __init__(
            self,
            data_reader,
            encoding_type="1D"
            ):
        
        self.data_reader = data_reader
        self.encoding_type = encoding_type

    def read_data(self,data_file,scale="MinMaxScaler"):

        # if scale < 1:
        #     print("Invalid scale value, defaulting to 1")
        #     scale = 1

        # Loading data
        self.data_reader.load_file(data_file) # Loading the first data file

        if self.encoding_type == "1D":
             self.X = self.data_reader.encode_seq_1d() # Hot-one encoded sequences
        elif self.encoding_type == "2D":
            self.X = self.data_reader.encode_seq_2d() # Counts of amino acids
        else:
            print("Encoding must be '1D' or '2D'. Defaulting to '1D'")
            self.X = self.data_reader.encode_seq_1d() # Hot-one encoded sequences

        # Abundance data
        self.y_abundance = self.data_reader.get_abundance()
        
        # Activity data
        self.y_activity  = self.data_reader.get_activity()

        # Since we have a single feature, we reshape as follows:
        self.y_abundance = self.y_abundance.values.reshape(-1, 1)
        self.y_activity = self.y_activity.values.reshape(-1, 1)
 
        # Scale the data by user given value
        scaler = MinMaxScaler(feature_range=(0, 1))
        self.y_abundance = scaler.fit_transform(self.y_abundance)
        self.y_activity = scaler.fit_transform(self.y_activity)
        # self.y_abundance = self.y_abundance / scale
        # self.y_activity = self.y_activity / scale
        
        size = len(self.y_abundance)
        
        return self.X, self.y_abundance, self.y_activity, size
    
    def split_one(self, df, overfit = False):
        first_split = .6
        second_split = first_split + (1-first_split)/2
        #train = df[:round(first_split* len(df))]
        #validate = df[round(first_split* len(df)):round(second_split*len(df))]
        #test = df[round(second_split* len(df)):]

        # Try to overfit the model to one data point
        train = df[:round(first_split*len(df))]
        if overfit:
            train = df[0:1]
            print(train)
        validate = df[round(first_split*len(df)):round(len(df)*second_split)]
        test = df[round(second_split*len(df)):]
        return train, validate, test


    def split_data(self, ratio, datasize = 0, overfit = False):

        # Added to be able to train on increasing number of samples
        if datasize ==0:
            datasize = len(self.X)

        # Create random permutation of indicies
        random.seed(525)
        index = [x for x in range(len(self.X))]
        random.shuffle(index)
        self.ratio = ratio

        # Reorder data based on shuffled indicies
        reordered_X = self.X[index][:datasize]
        # reordered_y_activity = self.y_activity.iloc[index][:datasize]
        # reordered_y_abundance = self.y_abundance.iloc[index][:datasize]
        reordered_y_activity = self.y_activity[index,:datasize]
        reordered_y_abundance = self.y_abundance[index,:datasize]

        # 60% of data for training, 20% for validating, 20% for testing
        train_X, val_X, test_X = self.split_one(reordered_X,overfit)
        
        self.tensor_X_train = torch.from_numpy(train_X).type(torch.FloatTensor)
        self.tensor_X_val = torch.from_numpy(val_X).type(torch.FloatTensor)
        self.tensor_X_test = torch.from_numpy(test_X).type(torch.FloatTensor)

        # Split Y
        train_Y_act, val_Y_act, test_Y_act = self.split_one(reordered_y_activity, overfit)
        train_Y_abund, val_Y_abund, test_Y_abund = self.split_one(reordered_y_abundance, overfit)


        if self.ratio:
            # Creating ratio (what the model predicts)
            y_train = train_Y_act / train_Y_abund # Making it into a percentage for higher loss values
            y_val = val_Y_act / val_Y_abund
            y_test = test_Y_act  / test_Y_abund

            # Removed .values from all of these
            self.tensor_y_train = torch.tensor(y_train).unsqueeze(1).type(torch.FloatTensor) #.unsqueeze(1) # makes the dimensions match: (18,1) instead of (18)
            self.tensor_y_val = torch.tensor(y_val).unsqueeze(1).type(torch.FloatTensor) 
            self.tensor_y_test = torch.tensor(y_test).unsqueeze(1).type(torch.FloatTensor)

        else:
            tensor_y_train_abund = torch.tensor(train_Y_abund).unsqueeze(1).type(torch.FloatTensor) #.unsqueeze(1) # makes the dimensions match: (18,1) instead of (18)
            tensor_y_val_abund = torch.tensor(val_Y_abund).unsqueeze(1).type(torch.FloatTensor)
            tensor_y_test_abund = torch.tensor(test_Y_abund).unsqueeze(1).type(torch.FloatTensor)

            tensor_y_train_act = torch.tensor(train_Y_act).unsqueeze(1).type(torch.FloatTensor) #.unsqueeze(1) # makes the dimensions match: (18,1) instead of (18)
            tensor_y_val_act = torch.tensor(val_Y_act).unsqueeze(1).type(torch.FloatTensor)
            tensor_y_test_act = torch.tensor(test_Y_act).unsqueeze(1).type(torch.FloatTensor)

            self.tensor_y_train = torch.stack((tensor_y_train_abund, tensor_y_train_act)).transpose(0,1)
            self.tensor_y_val = torch.stack((tensor_y_val_abund, tensor_y_val_act)).transpose(0,1)
            self.tensor_y_test = torch.stack((tensor_y_test_abund, tensor_y_test_act)).transpose(0,1)


    def read_split_data(self, train_file, val_file, test_file, scaler="MinMaxScaler", return_val=False, ratio=False):
        # Read in both the data somehow
        # Use scalar.fix to get min and max of all data
        # Transfor both data separately 

        # Reading in the training data
        self.data_reader.load_file(train_file) # Loading the first data file

        if self.encoding_type == "1D":
             self.X_train = self.data_reader.encode_seq_1d() # Hot-one encoded sequences
        elif self.encoding_type == "2D":
            self.X_train = self.data_reader.encode_seq_2d() # Counts of amino acids
        else:
            print("Encoding must be '1D' or '2D'. Defaulting to '1D'")
            self.X_train = self.data_reader.encode_seq_1d() # Hot-one encoded sequences

        self.y_train_abundance = self.data_reader.get_abundance()
        self.y_train_activity  = self.data_reader.get_activity()
        self.y_train_abundance = self.y_train_abundance.values.reshape(-1, 1)
        self.y_train_activity = self.y_train_activity.values.reshape(-1, 1)
        
        if ratio: 
            self.y_train_ratio = self.data_reader.get_ratio()
            self.y_train_ratio = self.y_train_ratio.values.reshape(-1, 1)

        # Reading in the validation data
        self.data_reader.load_file(val_file) # Loading the first data file

        if self.encoding_type == "1D":
             self.X_val = self.data_reader.encode_seq_1d() # Hot-one encoded sequences
        elif self.encoding_type == "2D":
            self.X_val = self.data_reader.encode_seq_2d() # Counts of amino acids
        else:
            print("Encoding must be '1D' or '2D'. Defaulting to '1D'")
            self.X_val = self.data_reader.encode_seq_1d() # Hot-one encoded sequences

        self.y_val_abundance = self.data_reader.get_abundance()
        self.y_val_activity  = self.data_reader.get_activity()
        self.y_val_abundance = self.y_val_abundance.values.reshape(-1, 1)
        self.y_val_activity = self.y_val_activity.values.reshape(-1, 1)

        if ratio: 
            self.y_val_ratio = self.data_reader.get_ratio()
            self.y_val_ratio = self.y_val_ratio.values.reshape(-1, 1)

        # Reading in the test data
        self.data_reader.load_file(test_file) # Loading the first data file

        if self.encoding_type == "1D":
             self.X_test = self.data_reader.encode_seq_1d() # One-hot encoded sequences
        elif self.encoding_type == "2D":
            self.X_test = self.data_reader.encode_seq_2d() # Counts of amino acids
        else:
            print("Encoding must be '1D' or '2D'. Defaulting to '1D'")
            self.X_test = self.data_reader.encode_seq_1d() # One-hot encoded sequences

        self.y_test_abundance = self.data_reader.get_abundance()
        self.y_test_activity  = self.data_reader.get_activity()
        self.y_test_abundance = self.y_test_abundance.values.reshape(-1, 1)
        self.y_test_activity = self.y_test_activity.values.reshape(-1, 1)

        if ratio: 
            self.y_test_ratio = self.data_reader.get_ratio()
            self.y_test_ratio = self.y_test_ratio.values.reshape(-1, 1)

        # Scaling the data
        if scaler == "1000":
            self.y_train_abundance = self.y_train_abundance / 1000
            self.y_val_abundance = self.y_val_abundance / 1000
            self.y_test_abundance = self.y_test_abundance / 1000

            self.y_train_activity = self.y_train_activity / 1000
            self.y_val_activity = self.y_val_activity / 1000
            self.y_test_activity = self.y_test_activity / 1000


        else:
            if scaler == "MinMaxScaler":
                scaler_abund = MinMaxScaler(feature_range=(0, 1))
                scaler_activity = MinMaxScaler(feature_range=(0,1))
                if ratio: 
                    scaler_ratio = MinMaxScaler(feature_range = (0,1))
            elif scaler == "StandardScaler":
                scaler_abund = StandardScaler()
                scaler_activity = StandardScaler()
                if ratio: 
                    scaler_ratio = StandardScaler()
            else: 
                scaler_abund = MinMaxScaler(feature_range=(0, 1))
                scaler_activity = MinMaxScaler(feature_range=(0,1))
                if ratio: 
                    scaler_ratio = MinMaxScaler(feature_range=(0,1))

            # print(self.y_train_abundance.flatten())
            all_abund = np.concatenate((self.y_train_abundance, self.y_val_abundance, self.y_test_abundance))
            all_activity = np.concatenate((self.y_train_activity, self.y_val_activity, self.y_test_activity))
            if ratio:
                all_ratio = np.concatenate((self.y_train_ratio, self.y_val_ratio, self.y_test_ratio))
            
            # Get min and max from all data
            if return_val:
                scaler_abund.fit(self.y_train_abundance)
                scaler_activity.fit(self.y_train_activity)
                if ratio:
                    scaler_ratio.fit(self.y_train_ratio)
            else:   
                scaler_abund.fit(all_abund)
                scaler_activity.fit(all_activity)
                if ratio: 
                    scaler_ratio.fit(all_ratio)

            dump(scaler_abund, 'scaler_abund.bin', compress=True)
            dump(scaler_activity, 'scaler_activity.bin', compress=True)
            if ratio: 
                dump(scaler_ratio, 'scaler_ratio.bin', compress=True)

            self.y_train_abundance = scaler_abund.transform(self.y_train_abundance)
            self.y_val_abundance = scaler_abund.transform(self.y_val_abundance)
            self.y_test_abundance = scaler_abund.transform(self.y_test_abundance)

            self.y_train_activity = scaler_activity.transform(self.y_train_activity)
            self.y_val_activity = scaler_activity.transform(self.y_val_activity)
            self.y_test_activity = scaler_abund.transform(self.y_test_activity)
            
            if ratio:
                self.y_train_ratio = scaler_ratio.transform(self.y_train_ratio)
                self.y_val_ratio = scaler_ratio.transform(self.y_val_ratio)
                self.y_test_ratio = scaler_ratio.transform(self.y_test_ratio)

        # Make everything into tensors
        self.tensor_X_train = torch.from_numpy(self.X_train).type(torch.FloatTensor)
        self.tensor_X_val = torch.from_numpy(self.X_val).type(torch.FloatTensor)
        self.tensor_X_test = torch.from_numpy(self.X_test).type(torch.FloatTensor)

        tensor_y_train_abund = torch.tensor(self.y_train_abundance).unsqueeze(1).type(torch.FloatTensor) #.unsqueeze(1) # makes the dimensions match: (18,1) instead of (18)
        tensor_y_val_abund = torch.tensor(self.y_val_abundance).unsqueeze(1).type(torch.FloatTensor)
        tensor_y_test_abund = torch.tensor(self.y_test_abundance).unsqueeze(1).type(torch.FloatTensor)

        tensor_y_train_act = torch.tensor(self.y_train_activity).unsqueeze(1).type(torch.FloatTensor) #.unsqueeze(1) # makes the dimensions match: (18,1) instead of (18)
        tensor_y_val_act = torch.tensor(self.y_val_activity).unsqueeze(1).type(torch.FloatTensor)
        tensor_y_test_act = torch.tensor(self.y_test_activity).unsqueeze(1).type(torch.FloatTensor)

        if ratio:
            tensor_y_train_ratio = torch.tensor(self.y_train_ratio).unsqueeze(1).type(torch.FloatTensor)
            tensor_y_val_ratio = torch.tensor(self.y_val_ratio).unsqueeze(1).type(torch.FloatTensor)
            tensor_y_test_ratio = torch.tensor(self.y_test_ratio).unsqueeze(1).type(torch.FloatTensor)

        if ratio:
            self.tensor_y_train = torch.stack((tensor_y_train_abund, tensor_y_train_act, tensor_y_train_ratio)).transpose(0,1)
            self.tensor_y_val = torch.stack((tensor_y_val_abund, tensor_y_val_act, tensor_y_val_ratio)).transpose(0,1)
            self.tensor_y_test = torch.stack((tensor_y_test_abund, tensor_y_test_act, tensor_y_test_ratio)).transpose(0,1)
        else:
            self.tensor_y_train = torch.stack((tensor_y_train_abund, tensor_y_train_act)).transpose(0,1)
            self.tensor_y_val = torch.stack((tensor_y_val_abund, tensor_y_val_act)).transpose(0,1)
            self.tensor_y_test = torch.stack((tensor_y_test_abund, tensor_y_test_act)).transpose(0,1)

        # full_analysis.py uses all data combined
        self.X = np.concatenate((self.X_train, self.X_val, self.X_test))
        self.y_abundance = np.concatenate((self.y_train_abundance,self.y_val_abundance,self.y_test_abundance))
        self.y_activity = np.concatenate((self.y_train_activity, self.y_val_activity,self.y_test_activity))
        size = len(self.y_abundance)
        
        if return_val:
            return self.X_val, self.y_val_abundance, self.y_val_activity, size
        return self.X, self.y_abundance, self.y_activity, size
    
    
    def load_data(self,batch_size):

        # Load data and set batch size
        train_data_loader = FastTensorDataLoader( 
                self.tensor_X_train, self.tensor_y_train, batch_size=batch_size, shuffle=True)
        
        val_data_loader = FastTensorDataLoader( 
                self.tensor_X_val, self.tensor_y_val, batch_size=batch_size, shuffle=True)
        
        test_data_loader = FastTensorDataLoader( 
                self.tensor_X_test, self.tensor_y_test, batch_size=batch_size, shuffle=True)
        
        if self.encoding_type == "2D":
            size = tuple(self.tensor_X_train.shape[1:4]) # Get second two dimensions (first dimension is # of samples)
        else:
            size = [self.tensor_X_train.shape[1]]

        # Size needs to be a tuple
        return train_data_loader, val_data_loader, test_data_loader, size


# Adapted from MoCHI
class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """
    def __init__(
        self, 
        *tensors, 
        batch_size = 32, 
        shuffle = False):
        """
        Initialize a FastTensorDataLoader.

        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :returns: FastTensorDataLoader object.
        """

        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches
    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i:self.i+self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches