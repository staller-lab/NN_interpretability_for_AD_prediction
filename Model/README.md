The code used to create and train the NNs. 

### Explanation of scripts
- `Model.py` : This file is the main file used to train the model. 
- `Data.py` : This file contains classes used to load the data for model training. 
- `ADModel_act.py` : This file contains the SimpleNN for activity class
- `ADModel_abund.py` : This file contains the SimpleNN for abundance class
- `ADModel_two_state.py` : This file contains the classes for the two-state BiophysicalNN. One class permits a variable sized abundance convolutional filter while the other does not. 
- `ADModel_three_state.py` : This file contains the classes for the three-state BiophysicalNN.  One class permits a variable sized abundance convolutional filter while the other does not. 

### Parameters for `Model.py`: 
- `k` --> size of convolutional filter
- `wp` --> how much to penalize negative weights
- `seed` --> seed for random initalization
- `e` --> epochs
- `c` --> channels of the convolution filter (I've only played around with one channel, more than 1 becomes very uninterpretable)
- `m` --> key word for the type of model (simple_abund, simple_act, three_state_abund, two_state_abund, three_state, two_state. Three_state_abund and two_state_abund allow you to set the size of the abundance convolution filter separately from the other convolutional filters using the argument -ak) 
- `np` --> how much to penalize negative activity predictions
- `s` --> how to scale the input data, I always used MinMaxScaler
- `i` --> whether to use a preditermined data split (passed in by the user)
- `f` --> the input file, if argument i is passed, this should only be the training data
- `v` --> the validation data file
- `t` --> the test data file
- `hv` --> the hill coefficient to use in biophysical models
- `out_model` --> name of the output model
- `b` --> the batch size, I used 10
- `l` --> the learning rate, 0.001 worked best
- `a` --> Transformation to go from amount of bound TF to GFP in the BiophysicalNNs. The two options are linear or Hill.
- `r` --> Whether to use a ReLU activation function. If not set, will use Parametric ReLU (what I used)

**NOTE: The actual commands used to train the models can be found in the `scripts/trainning_scripts` folder.**  

