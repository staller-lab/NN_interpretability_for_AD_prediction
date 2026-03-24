results_file="../../results/results_three_state_bootstraps_leakyrelu.csv"
model=three_state_abund
e=200
np=0.5
wp=0.1

A=Hill
hill_val=1

for i in {0..199}; 
do 
    train_file=../../Data/bootstraps/pm_gcn4_sort2_pools_allchannels_wrangled_w_ratio_training_bootstrapped_${i}.csv
    val_file=../../Data/pm_gcn4_sort2_pools_allchannels_wrangled_w_ratio_validation_heirarchical_v2.csv
    test_file=../../Data/pm_gcn4_sort2_pools_allchannels_wrangled_w_ratio_test_heirarchical_v2.csv

    for seed in 1298091;
    do for ak in 30; #5 10 15 20 30 40;
    do for K in 5 10 15 20 30 40;
    do
        out_model=../../model_data/${model}model_b10_e${e}_L0.001_sMinMax_a${A}_c1_k${K}_ak${ak}_LeakyReLU_${np}PosKLoss_${wp}PosWeightLoss_hSplit_hv${hill_val}_seed${seed}_weightInit_test_BOOTSTRAP_${i}
        echo $out_model

        python ../../Model/Model.py -f $train_file -v $val_file -t $test_file -i -b 10 -l 0.001 -e $e -s MinMaxScaler -a $A -c 1 -k $K -ak $ak -o $out_model -m $model -np $np -wp $wp -hv $hill_val -seed $seed
        python make_summary_file.py -np $np -wp $wp -f $train_file -v $val_file -t $test_file -i -s MinMaxScaler -a $A -c 1 -k $K -ak $ak -m $out_model -n $model -hv $hill_val -rf $results_file

done;
done;
done; 
done; 
