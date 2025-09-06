"""
Simple script to read in activity data and produce csv file
ready to be used in training. Same as data_rangle.py but 
does not average measurements from reps, instead keeping 
them as separate entries. 

Usage: python data_rangle_reps.py [File path]

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

data_A = { 'aa_seq' : prot_seq, 'activity' : file["Activity_gfpA"], 'abundance' : file["Activity_cherryA"]}
df_A = pd.DataFrame(data_A).dropna()

data_B = { 'aa_seq' : prot_seq, 'activity' : file["Activity_gfpB"], 'abundance' : file["Activity_cherryB"]}
df_B = pd.DataFrame(data_B).dropna()


df = pd.concat([df_A, df_B], axis = 0)

# Use regular expression to extract the "filename" part
match = re.match(r'(.+)(\..+)', filename)

if match:
    filename_without_extension = match.group(1)
else:
    # Handle the case where there's no match (e.g., if the input is not a valid filename)
    filename_without_extension = None

df.to_csv(filename_without_extension + "_wrangled_reps.csv",index=False)
