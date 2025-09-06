"""
Simple script to read in activity data and produce csv file
ready to be used in training. Same as data_rangle.py but 
removes max values (as they may be saturated). 

Usage: python data_rangle_drop_max.py [File path]

Written by Claire LeBlanc
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
activity.fillna(file["Activity_gfpB"], inplace=True)
activity.fillna(file["Activity_gfpA"], inplace=True) 

abundance = (file["Activity_cherryA"]  + file["Activity_cherryB"]) / 2
abundance.fillna(file["Activity_cherryB"], inplace=True)
abundance.fillna(file["Activity_cherryA"], inplace=True) 


data = {'aa_seq': prot_seq, 'activity':activity, 'abundance':abundance}
df = pd.DataFrame(data).dropna()


                                      
# Use regular expression to extract the "filename" part
match = re.match(r'(.+)(\..+)', filename)

if match:
    filename_without_extension = match.group(1)
else:
    # Handle the case where there's no match (e.g., if the input is not a valid filename)
    filename_without_extension = None


df.to_csv(filename_without_extension + "_wrangled.csv",index=False)
