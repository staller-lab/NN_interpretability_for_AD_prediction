
# Explanation of scripts: 


`full_analysis.py` --> generated summary plots for a given model. Scatter plots showing correlations and interpretation plots visualizing the parameters of the NN. 
`make_summary_file.py` --> writes summary stats (correlation coefficients, hyperparameters, etc) to a file. Either creates or appends to the results.csv file. 

`train_*.sh` --> Contains the command line code to train the various versions of model, generate summary figures and write the results to a summary file (will append the correlation coefficients and data on the training run to results.csv). There are many different command line options to determine the hyperparameters of the mode. 

### Parameters for `full_analysis.py` and `make_summary_file.py`
Very similar to `Model.py`, with the following differences: 
- `o` --> this is now the location of the output folder where to save the generated figures
- `m` --> this is now the name of the saved model from Model.py
- `n` --> this is now the name of the type of model (simple_abund, simple_act, etc.)


