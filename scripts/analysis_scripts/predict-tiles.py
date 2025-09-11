from finches.utils import folded_domain_utils
from finches import CALVADOS_frontend, Mpipi_frontend
import pandas as pd
import numpy as np
from Bio.Seq import Seq

def predict():
    #setup folded domain
    abd1 = folded_domain_utils.FoldeDomain('../../data/abd1.pdb')
    
    #setup parameters
    ff = CALVADOS_frontend()
    window_size = 15
    
    #import seq data
    tiles_df = pd.read_csv('../../data/pm_gcn4_sort2_pools_allchannels.csv', index_col=0)
    tiles_df['Seq'] = tiles_df.apply(lambda row: str(Seq(row['ArrayDNA']).translate()), axis=1)
    
    #predict
    tiles_df['result'] = tiles_df.apply(lambda row: predict_tiles(row['Seq'], abd1, ff, window_size), axis=1)
    tiles_df['attractive'] = tiles_df['result'].str.get(0)
    tiles_df['repulsive'] = tiles_df['result'].str.get(1)
    tiles_df['self'] = tiles_df['result'].str.get(2)
    tiles_df['average'] = tiles_df['result'].str.get(3)
    
    #save
    tiles_df.to_csv('./predict-tiles.csv')
    
def predict_tiles(seq, folded_domain, ff, window_size=15):
    '''
    Args:
        - seq (str): sequence of IDR tile
        - folded_domain (finches.folded_domain_utils.FoldedDomain): preprocessed folded domain object for
                                                                    PDB of scMed15-ABD1 (158-236AA)
        - ff (): frontend object for either CALVDOS or Mpipi force field
        - window_size (int): window size to scan over IDR to calculate IDR-patch interactions
    '''
    X = folded_domain.calculate_idr_surface_patch_interactions(seq, ff.IMC_object, window_size)
    matrix = X[1]

    #1. Mask only attractive/repulsive interactions by < or > 0
    #2. Take mean of matrix to get mean-field term
    #3. In case entire field is NaN (all are pos/neg), imputes to 0
    attractive = np.mean(matrix[matrix < 0]) if matrix[matrix < 0].size > 0 else 0
    repulsive = np.mean(matrix[matrix > 0]) if matrix[matrix > 0].size > 0 else 0
    average = np.nanmean(matrix) if np.isnan(matrix).any() else np.mean(matrix)

    #self-attractive
    X = ff.intermolecular_idr_matrix(seq, seq, window_size=window_size)
    matrix = X[0][0]
    self_attractive = np.nanmean(matrix) if np.isnan(matrix).any() else np.mean(matrix)
    
    return attractive, repulsive, self_attractive, average

def main():
    predict()

if __name__ == "__main__":
    main()
