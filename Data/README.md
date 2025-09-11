## Explanation of scripts

The scripts that were actually used in the final version of the paper are: 

- `data_rangle_w_ratio.py` : This is the code used to generate the inital training data frame. Takes in data file with multiple reps and averages them. If only one rep is present, that rep is maintained. If a sequence does not have measurement for GFP and mCherry it is removed.
  - The other data_rangle*.py files are old and left over from various things I tried
- `split_data_w_test.ipynb` : This is the code that generates the train/validation/test spilt used in training. Heirarchically clusters the full length sequences, maps tiles to full length sequences, and writes output to three files.
  - The other split_data.ipynb is from before.
  - Contains code to create **Figure S14**

  
