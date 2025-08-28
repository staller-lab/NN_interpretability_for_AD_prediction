Code used to perform the analyses described in **Simple and interpretable neural networks of transcriptional activation domains separate roles of protein abundance and coactivator binding**

### Neural network architecture and trainning
The code used to define and train the models is located in scripts folder in the `Model.py` file. The code used to prepare the data for trainning is located in the `Data.py` file. 

The shell scripts used to train the model are also located in the scripts folder: 
- `train_simple_abund.sh` --> Trains the SimpleNN-abund
- `train_simple_act.sh` --> Trains the SimpleNN-GFP
- `train_two_state_abund.sh` --> Trains the two state BiophysicalNN
- `train_three_state_abund.sh` --> Trains the three state BiophysicalNN

### Analysis scripts
All of the scripts used to analyze the NNs and create the figures are also located in the scripts folder. These are described in depth in that folder. 


