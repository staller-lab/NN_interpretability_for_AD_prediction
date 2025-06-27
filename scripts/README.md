
# Explanation of scripts: 

`Data.py` --> Loads the datasets
`Model.py` --> Contains classes for the simple and biophysical NNs, contains training code

`full_analysis.py` --> generated summary plots for a given model. Scatter plots showing correlations and interpretation plots visualizing the parameters of the NN. 
`make_summary_file.py` --> writes summary stats (correlation coefficients, hyperparameters, etc) to a file. Either creates or appends to the results.csv file. 

`train_*.sh` --> Contains the command line code to train the various versions of model, generate summary figures and write the results to a summary file (will append the correlation coefficients and data on the training run to results.csv). There are many different command line options to determine the hyperparameters of the mode. 

### Parameters for `Model.py`: 
- `k` --> size of convolutional filter
- `wp` --> how much to penalize negative weights
- `seed` --> seed for random initalization
- `e` --> epochs
- `c` --> channels of the convolution filter (I've only played around with one channel, more than 1 becomes very uninterpretable)
- `m` --> key word for the type of model (simple_abund, simple_act, three_state_abund, two_state_abund, three_state, two_state. Three_state_abund and two_state_abund allow you to set the size of the abundance convolution filter separately from the other convolutional filters using the argument -ak) 
- `ap` --> how much to weight activity vs. abundance in training, no longer relevant because they are getting trainned separately
- `np` --> how much to penalize negative activity predictions
- `s` --> how to scale the input data, I always used MinMaxScaler
- `i` --> whether to use a preditermined data split (passed in by the user)
- `f` --> the input file, if argument i is passed, this should only be the trainning data
- `v` --> the validation data file
- `hv` --> the hill coefficient to use in biophysical models
- `out_model` --> name of the output model

### Parameters for `full_analysis.py` and `make_summary_file.py`
Very similar to `Model.py`, with the following differences: 
- `o` --> this is now the location of the output folder where to save the generated figures
- `m` --> this is now the name of the saved model from Model.py
- `n` --> this is now the name of the type of model (simple_abund, simple_act, etc.)


### Other notebooks to explore the results
- `random_seed_explore.ipynb`:
	- Code to create:
		- Supplemental Figure 1
		- Supplemental Figure 4
		- Supplemental Figure 5
- `abundance_replicate_correlation.ipynb`: Determines how correlated experimental mCherry measurements are
- `compare_bound_transformations.ipynb`: Compares functions (Linear vs. Hill) to transform bound TF to GFP
- `compare_simple_w_deep_abund.ipynb`:
  	- Code to create:
  		- Figure 2 plots
- `compare_simple_w_deep_abund.ipynb`:
	- Code to create:
 		- Figure 1 plots
- `compare_two_and_three_states.ipynb`:
	- Code to create:
 		- Figure 3 and 4 plots
- `comparing_seeds.ipynb`:
	- Code to create supplemental Figures X and X
- `degron_screen_data.ipynb`:
 	- Analysis of data from Larsen et al 2025
- `linear_regression.ipynb`:
  	- Linear regression on the activity data
-  `seed_explore.ipynb`
-  `split_data.ipynb`: Code to split the training and validation data for minimal overlap
-  `test_paddle.ipynb`
-  `TF_endings.ipynb`: Code to analyze the terminal amino acids for YGOB homologous sequences
 
