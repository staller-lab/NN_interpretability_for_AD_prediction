results_file="../../results/results_simple_shuffles_leakyrelu.csv"
model=simple_act
e=200
wp=0.5

for i in {400..499}; 
do 
    train_file=../../Data/shuffles/pm_gcn4_sort2_pools_allchannels_wrangled_w_ratio_training_shuffled_${i}.csv
    val_file=../../Data/shuffles/pm_gcn4_sort2_pools_allchannels_wrangled_w_ratio_validation_shuffled_${i}.csv
    test_file=../../Data/shuffles/pm_gcn4_sort2_pools_allchannels_wrangled_w_ratio_test_shuffled_${i}.csv

    for seed in 3297031; #830655019 8923039 3297062 1793115 27914621 1834636 7468236 82376482 5273682 1298091;
    do for K in 5; # 10 15 20 30 40;
    do
        out_model=../../model_data/${model}model_b10_e${e}_L0.001_sMinMax_c1_k${K}_wp${wp}_LeakyReLU_hSplit_seed${seed}_weightInit_test_SHUFFLE_${i}
        out_folder=../../results/${model}model_b10_e${e}_L0.001_sMinMax_c1_k${K}_wp${wp}_LeakyReLU_hSplit_seed${seed}_weightInit_test_SHUFFLE_${i}
        if [ -f "$out_model.pth" ]; then
            echo $out_model
        else
            python ../../Model/Model.py -f $train_file -v $val_file -t $test_file -i -b 10 -l 0.001 -e $e -s MinMaxScaler -c 1 -k $K -o $out_model -m $model -wp $wp -seed $seed
            # rm -rf $out_folder
            # python full_analysis.py -f $train_file -v $val_file  -i -s MinMaxScaler -c 1 -k $K -o $out_folder -m $out_model -n  $model
            python make_summary_file.py -f $train_file -v $val_file -t $test_file -i -s MinMaxScaler -c 1 -k $K -m $out_model -n  $model -wp $wp -rf $results_file
        fi
done;
done;
done; 
