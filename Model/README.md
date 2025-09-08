The code used to create and train the NNs. 

`Model.py` : This file is the main file used to train the model. It can be called with XXX. It initalizes the model, loads the data, and trains the model. 


- `Data.py` : This file contains classes used to load the data for model training. 
- `ADModel_act.py` : This file contains the SimpleNN for activity class
- `ADModel_abund.py` : This file contains the SimpleNN for abundance class
- `ADModel_two_state.py` : This file contains the classes for the two-state BiophysicalNN. One class permits a variable sized abundance convolutional filter while the other does not. 
- `ADModel_three_state.py` : This file contains the classes for the three-state BiophysicalNN.  One class permits a variable sized abundance convolutional filter while the other does not. 
