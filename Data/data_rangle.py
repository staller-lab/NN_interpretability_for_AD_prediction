"""
Simple script to read in activity data and produce csv file
ready to be used in training. 

Usage: python data_rangle.py [File path]

Written by Claire LeBlanc
Last modified 10/2/23
"""

import sys
import pandas as pd
from Bio.Seq import Seq
import re

# Quickly read file
# Check if the script is called with at least one argument
if len(sys.argv) < 3:
    print("Usage: python script.py <file path> ")
    sys.exit(1)

filename = sys.argv[1]

file = pd.read_csv(filename)

prot_seq = [Seq(seq).translate() for seq in file["ArrayDNA"]]

prot_seqs = pd.concat([prot_seq,prot_seq],axis=0)
activity = pd.concat([file["Activity_gfpA"],file["Activity_gfpB"]], axis = 0)
abundance = pd.concat([file["Activity_cherryA"],file["Activity_cherryB"]], axis =0)


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
