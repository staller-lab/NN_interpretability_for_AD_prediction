import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

filename = "all_seqs_overlap_matrix.csv"
data_df = pd.read_csv(filename, index_col=0)
data_df = data_df.drop(data_df.index[17323])
num = int(sys.argv[1])

# if trying to split an already split group
for i in range(num-1):
    group = pd.read_csv(f"group_{i+1}_seqs.csv", header=None)
    data_df = data_df.loc[[i not in group.values for i in data_df.index], [i not in group.values for i in data_df.columns]]

seq1 = data_df.index[0]
group = set()
queue = []
queue.append(seq1)

for i in queue:
    seq = queue.pop(0)
    if seq in group:
        continue
    for seq2 in data_df.loc[seq].index:
        i = data_df.loc[seq][seq2]
        if int(i) > 5:
            if seq2 not in queue:
                queue.append(seq2)
    group.add(seq)
        
with open(f"group_{num}_seqs.csv", "w+") as f:
    for i in group:
        f.write(i+"\n")
