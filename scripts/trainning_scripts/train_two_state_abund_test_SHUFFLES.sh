A=Hill
np=0.5
wp=0.1
hill_val=1
# CHANGE THIS!
results_file="../../results/results_two_state_shuffles_hill.csv"

for i in {400..499}; 
do 
    train_file=../../Data/shuffles/pm_gcn4_sort2_pools_allchannels_wrangled_w_ratio_training_shuffled_${i}.csv
    val_file=../../Data/shuffles/pm_gcn4_sort2_pools_allchannels_wrangled_w_ratio_validation_shuffled_${i}.csv
    test_file=../../Data/shuffles/pm_gcn4_sort2_pools_allchannels_wrangled_w_ratio_test_shuffled_${i}.csv

    model=two_state_abund
    e=200

    for seed in 1298091; #830655019 8923039 3297062 1793115 27914621 1834636 7468236 82376482 5273682 1298091;
    do for ak in 30; #5 10 15 20 30 40;
    do for K in 40; #5 10 15 20 30; # 40;
    do

        # CHANGE
        out_model=../../model_data/${model}model_b10_e${e}_L0.001_sMinMax_a${A}_c1_k${K}_ak${ak}_LeakyReLU_${np}PosKLoss_${wp}PosWeightLoss_hSplit_hv${hill_val}_seed${seed}_weightInit_SHUFFLE_${i}
        out_folder=../../results/${model}model_b10_e${e}_L0.001_sMinMax_a${A}_c1_k${K}_ak${ak}_LeakyReLU_${np}PosKLoss_${wp}PosWeightLoss_hSplit_hv${hill_val}_seed${seed}_weightInit_SHUFFLE_${i}
        if [ -f "$out_model.pth" ]; then
            echo $out_model
        else
            python ../../Model/Model.py -f $train_file -v $val_file -t $test_file -i -b 10 -l 0.001 -e $e -s MinMaxScaler -a $A -c 1 -k $K -ak $ak -o $out_model -m $model -np $np -wp $wp -hv $hill_val -seed $seed
            # rm -rf $out_folder
            # python full_analysis.py -f $train_file -v $val_file -t $test_file -i -s MinMaxScaler -a $A -c 1 -k $K -ak $ak -o $out_folder -m $out_model -n  $model -hv $hill_val
            python make_summary_file.py -np $np -wp $wp -f $train_file -v $val_file -t $test_file -i -s MinMaxScaler -a $A -c 1 -k $K -ak $ak -m $out_model -n  $model -hv $hill_val -rf $results_file
        fi

done;
done;
done;
done; 
