This is the code used to perform the analyses found in the paper, including to create all of the figures in the paper. Brief describtion of each notebook is found below. 

### Main analysis notebooks
- `compare_simple_w_deep_activity.ipynb` : This notebook creates all the plots in Figure 1, specifically comparing the SimpleNN to deep NNs and visualizing the SimpleNN parameters.
- `compare_simple_w_deep_abund.ipynb` : This notebook creates all the plots in Figure 2, specifically comparing the SimpleNN abund to a deep NNs, visualizing the SimpleNN parameters, and plotting abundance histograms. Also makes histograms for supplemental figure 5. 
- `compare_two_and_three_state.ipynb` : This notebook creates the scatter plots in Figures 3 and 4. Also creates the parameter visualizations in Figures 2 and 3. 

### Supplemental analysis notebooks
- `random_seed_explore.ipynb`:
	- Code to create:
		- Supplemental Figure 1
		- Supplemental Figure 4
		- Supplemental Figure 5
- `abundance_replicate_correlation.ipynb`: Determines how correlated experimental mCherry measurements are
- `compare_bound_transformations.ipynb`: Compares functions (Linear vs. Hill) to transform bound TF to GFP
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
 
