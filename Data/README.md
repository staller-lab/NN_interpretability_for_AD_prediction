## Explanation of scripts

The scripts that were actually used in the final version of the paper are: 

- `data_rangle_w_ratio.py` : This is the code used to generate the inital training data frame. Takes in data file with multiple reps and averages them. If only one rep is present, that rep is maintained. If a sequence does not have measurement for GFP and mCherry it is removed.
  - The other data_rangle*.py files are old and left over from various things I tried
- `split_data_w_test.ipynb` : This is the code that generates the train/validation/test spilt used in training. Heirarchically clusters the full length sequences, maps tiles to full length sequences, and writes output to three files.
  - The other split_data.ipynb is from before.
  - Contains code to create **Figure S14**

  
## Explanation of data
- `pm_gcn4_sort2_pools_allchannels_wrangled_w_ratio.csv` : This is the data produced by `data_rangle_w_ratio.py`. Contains a column for the 40amino acid sequence (`aa_seq`), a column for the reporter gene activation (`activity`), and a column for the TF abundance (`abundance`).
- `pm_gcn4_sort2_pools_allchannels_wrangled_w_ratio_test_heirarchical_v2.csv` : This is the test data produced by `split_data_w_test.ipynb`. Test data was used in the final evaluation of the best model.  
- `pm_gcn4_sort2_pools_allchannels_wrangled_w_ratio_validation_heirarchical_v2.csv` : This is the validation data produced by `split_data_w_test.ipynb`. Validation data was used in early stopping and selecting the best model. 
- `pm_gcn4_sort2_pools_allchannels_wrangled_w_ratio_validation_heirarchical_v2.csv` : This is the training data produced by `split_data_w_test.ipynb`. This set of training data was used to train all the models. 
