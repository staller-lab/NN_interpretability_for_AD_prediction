This is the code used to perform the analyses found in the paper, including to create all of the figures in the paper. Brief describtion of each notebook is found below. 

### Main analysis notebooks
These notebooks create the majority of the main figure pannels. 
- `compare_simple_w_deep_activity.ipynb` : This notebook creates all the plots in **Figure 1**, specifically comparing the SimpleNN to deep NNs and visualizing the SimpleNN parameters.
- `compare_simple_w_deep_abund.ipynb` : This notebook creates all the plots in **Figure 2**, specifically comparing the SimpleNN abund to a deep NNs, visualizing the SimpleNN parameters, and plotting abundance histograms. Also makes histograms for Figure S5.
- `compare_two_and_three_state.ipynb` : This notebook creates the scatter plots in **Figures 3 and 4**. Also creates the parameter visualizations in **Figures 2 and 3**.
- `tile_replaced_analysis_Pooja.ipynb` : This notebook performs a post-hoc analysis of the deep neural networks. It loads predictions for all variants and summarizes them. Contains code to create **Figure 3H and S12**.
	- NEED TO FINALIZE

### Supplemental analysis notebooks
- `comparing_seeds.ipynb` : This notebook compares the performance of many different random initalizations of the SimpleNN and BiophysicalNNs. 
	- Contains code to create **Figure S1**
	- Contains code to create **Figure S9**
- `test_on_new_data.ipynb` : This notebook investigates the performance of simple, biophysical, and deep NNs on data from other screens.
	- Contains code to create **Figure S2**
- `random_seed_explore.ipynb`: This notebook creates figures to summarize the parameters of many NNs trained with different random initalizations.  
	- Specifically, contains code to create:
		- **Figure S3**
    	- **Figure S4**
		- **Figure S10**
		- **Figure S13**
- `degron_screen_data.ipynb` : Investigates data from two human degron screens from Larsen et al. (2025) and Voutsinos et al. (2025)
	- Contains code to create **Figure S6**
	- Revisit this! Need to finalize
 - `natural_yeast_ADs.ipynb` : Investigates whether there is differences in the amount of negative amino acids between C-terminal and non-C-terminal activation domains.
	- Contains code to create **Figure S7**
 - `equilibrium_constants.ipynb` : Explores the relationship between the equilibrium constant, the activity and the finches score.
	- Contains code to create **Figure S11**
- `compare_bound_transformations.ipynb`: Compares functions (Linear vs. Hill) to transform bound TF to GFP
	- Contains code to create **Figure S15**

### Other analysis notebooks (do not create figures in paper)
- `abundance_replicate_correlation.ipynb`: Determines how correlated experimental mCherry measurements are
- `linear_regression.ipynb`:
  	- Linear regression on the activity data
-  `TF_endings.ipynb`: Code to analyze the terminal amino acids for YGOB homologous sequences. These plots were not used in the final iteration of this paper. 
 
