A=Hill
np=0.1
wp=0.1
ap=1
hill_val=1
train_file=../../Data/pm_gcn4_sort2_pools_allchannels_wrangled_training_heirarchical.csv
val_file=../../Data/pm_gcn4_sort2_pools_allchannels_wrangled_validation_heirarchical.csv
model=three_state
e=200

for K in 4 5 6 7 8 9 10 11 12 13 14 15 20 30 40;
do
    out_model=../model_data/${model}model_b10_e${e}_L0.0001_sMinMax_a${A}_c1_k${K}_ParaReLU_${np}PosKLoss_${wp}PosWeightLoss_${ap}actLoss_hSplit_hv${hill_val}
    out_folder=../results/${model}model_b10_e${e}_L0.0001_sMinMax_a${A}_c1_k${K}_ParaReLU_${np}PosKLoss_${wp}PosWeightLoss_${ap}actLoss_hSplit_hv${hill_val}
    echo $out_model

    python Model.py -f $train_file -v $val_file  -i -b 10 -l 0.0001 -e $e -s MinMaxScaler -a $A -c 1 -k $K -o $out_model -m $model -np $np -wp $wp -ap $ap -hv $hill_val 
    rm -rf $out_folder
    python full_analysis.py -f $train_file -v $val_file  -i -s MinMaxScaler -a $A -c 1 -k $K -o $out_folder -m $out_model -n  $model -hv $hill_val
    python make_summary_file.py -np $np -wp $wp -f $train_file -v $val_file  -i -s MinMaxScaler -a $A -c 1 -k $K -m $out_model -n  $model -hv $hill_val -aw $ap

done

