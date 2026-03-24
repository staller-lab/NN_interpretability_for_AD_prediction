results_file="../../results/results_simple_abund_bootstraps_leakyrelu.csv"
model=simple_abund
e=200
wp=0.5

for i in {0..200}; 
do 
    train_file=../../Data/bootstraps/pm_gcn4_sort2_pools_allchannels_wrangled_w_ratio_training_bootstrapped_${i}.csv
    val_file=../../Data/pm_gcn4_sort2_pools_allchannels_wrangled_w_ratio_validation_heirarchical_v2.csv
    test_file=../../Data/pm_gcn4_sort2_pools_allchannels_wrangled_w_ratio_test_heirarchical_v2.csv

    for seed in 830654943;
    do for K in 5 10 15 20 30 40;
    do
        out_model=../../model_data/${model}model_b10_e${e}_L0.001_sMinMax_c1_k${K}_wp${wp}_LeakyReLU_hSplit_seed${seed}_weightInit_test_BOOTSTRAP_${i}
        echo $out_model

        python ../../Model/Model.py -f $train_file -v $val_file -t $test_file -i -b 10 -l 0.001 -e $e -s MinMaxScaler -c 1 -k $K -o $out_model -m $model -wp $wp -seed $seed
        python make_summary_file.py -f $train_file -v $val_file -t $test_file -i -s MinMaxScaler -c 1 -k $K -m $out_model -n  $model -wp $wp -rf $results_file

done;
done;
done; 
done; 
