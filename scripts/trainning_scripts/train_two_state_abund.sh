A=Linear
np=0.1
wp=0.5
ap=1
hill_val=1
train_file=../../../Data/pm_gcn4_sort2_pools_allchannels_wrangled_training_heirarchical.csv
val_file=../../../Data/pm_gcn4_sort2_pools_allchannels_wrangled_validation_heirarchical.csv
model=two_state_abund
ak=40
e=200

for seed in 830655019 8923039 3297062 1793115 27914621 1834636 7468236 82376482 5273682 1298091;
do for ak in 20 30 40;
do for K in 5 10 15 20 30 40;
do
    out_model=../../model_data/${model}model_b10_e${e}_L0.0001_sMinMax_a${A}_c1_k${K}_ak${ak}_ParaReLU_${np}PosKLoss_${wp}PosWeightLoss_hSplit_hv${hill_val}_seed${seed}_weightInit_v2
    out_folder=../../results/${model}model_b10_e${e}_L0.0001_sMinMax_a${A}_c1_k${K}_ak${ak}_ParaReLU_${np}PosKLoss_${wp}PosWeightLoss_hSplit_hv${hill_val}_seed${seed}_weightInit_v2
    echo $out_model

    python ../../Model/Model.py -f $train_file -v $val_file  -i -b 10 -l 0.0001 -e $e -s MinMaxScaler -a $A -c 1 -k $K -ak $ak -o $out_model -m $model -np $np -wp $wp -hv $hill_val -seed $seed
    rm -rf $out_folder
    python full_analysis.py -f $train_file -v $val_file  -i -s MinMaxScaler -a $A -c 1 -k $K -ak $ak -o $out_folder -m $out_model -n  $model -hv $hill_val
    python make_summary_file.py -np $np -wp $wp -f $train_file -v $val_file  -i -s MinMaxScaler -a $A -c 1 -k $K -ak $ak -m $out_model -n  $model -hv $hill_val -aw $ap

done;
done;
done;

