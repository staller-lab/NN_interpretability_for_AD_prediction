Code used to perform the analyses described in **Simple and interpretable biophysical neural networks of transcriptional activation domains separate roles of protein abundance and coactivator binding**

### Data
The code used to prepare the data for training and generate train/validation/test splits is in the `Data` folder. The datasets are also found there.

### Neural network architecture and training
The code used to load the data and create and train the models is located in `Model` folder. More information can be found there. 

The shell scripts used to train the model are also located in the `scripts/trainning_scripts` folder: 
- `train_simple_abund.sh` --> Trains the SimpleNN-abund
- `train_simple_act.sh` --> Trains the SimpleNN-GFP
- `train_two_state_abund.sh` --> Trains the two state BiophysicalNN
- `train_three_state_abund.sh` --> Trains the three state BiophysicalNN

### Analysis scripts
All of the scripts used to analyze the NNs and create the figures are also located in the `scripts/analysis_scripts`. These are described in depth in that folder. 

The code used to run DeepLift on ADHunter-ratio (Figure 3G) is located in this repository: https://github.com/staller-lab/ADHunter-DeepLift
