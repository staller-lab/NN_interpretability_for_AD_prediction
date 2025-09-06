"""
Simple script to read in activity data and produce csv file
ready to be used in training. Same as data_rangle.py but 
also includes ratio data as measurement in file. 

Usage: python data_rangle_w_ratio.py [File path]

Written by Claire LeBlanc
Last modified 10/2/23
"""

import sys
import pandas as pd
from Bio.Seq import Seq
import re

# Quickly read file
# Check if the script is called with at least one argument
if len(sys.argv) < 2:
    print("Usage: python script.py <file path>")
    sys.exit(1)

filename = sys.argv[1]

file = pd.read_csv(filename)

prot_seq = [Seq(seq).translate() for seq in file["ArrayDNA"]]

activity = (file["Activity_gfpA"]  + file["Activity_gfpB"]) / 2
activity.fillna(file["Activity_gfpB"], inplace=True) # Fills na values with gfpB activity if it exists
activity.fillna(file["Activity_gfpA"], inplace=True) # Fills na values with gfpA activity if it exists

abundance = (file["Activity_cherryA"]  + file["Activity_cherryB"]) / 2
abundance.fillna(file["Activity_cherryB"], inplace=True)
abundance.fillna(file["Activity_cherryA"], inplace=True) 

ratio = (file["Activity_ratioA"] + file["Activity_ratioB"]) / 2
ratio.fillna(file["Activity_ratioA"], inplace=True)
ratio.fillna(file["Activity_ratioB"], inplace=True)

data = {'aa_seq': prot_seq, 'activity':activity, 'abundance':abundance, 'ratio':ratio}
df = pd.DataFrame(data).dropna()

                                 
# Use regular expression to extract the "filename" part
match = re.match(r'(.+)(\..+)', filename)

if match:
    filename_without_extension = match.group(1)
else:
    # Handle the case where there's no match (e.g., if the input is not a valid filename)
    filename_without_extension = None


df.to_csv(filename_without_extension + "_wrangled_w_ratio.csv",index=False)
