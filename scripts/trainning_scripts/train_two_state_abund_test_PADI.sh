A=Hill
np=0.1
wp=0.5
hill_val=1

train_file="../../Data/PADI_data_wrangled_training_heirarchical.csv"
val_file="../../Data/PADI_data_wrangled_validation_heirarchical.csv"
test_file="../../Data/PADI_data_wrangled_test_heirarchical.csv"
results_file="../../results/PADI_results_two_state_test_leakyReLU.csv"

model=two_state_abund
e=200

for seed in 830655019 8923039 3297062 1793115 27914621 1834636 7468236 82376482 5273682 1298091;
do for ak in 1 2 3 4 5 10 15 20 25 30 35 40;
do for K in 1 2 3 4 5 10 15 20 25 30 35 40;
do
    out_model=../../model_data/PADI_${model}model_b10_e${e}_L0.001_sMinMax_a${A}_c1_k${K}_ak${ak}_LeakyReLU_${np}PosKLoss_${wp}PosWeightLoss_hSplit_hv${hill_val}_seed${seed}_weightInit
    out_folder=../../results/PADI_${model}model_b10_e${e}_L0.001_sMinMax_a${A}_c1_k${K}_ak${ak}_LeakyReLU_${np}PosKLoss_${wp}PosWeightLoss_hSplit_hv${hill_val}_seed${seed}_weightInit
    echo $out_model

    python ../../Model/Model.py -f $train_file -v $val_file -t $test_file -i -b 10 -l 0.001 -e $e -s MinMaxScaler -a $A -c 1 -k $K -ak $ak -o $out_model -m $model -np $np -wp $wp -hv $hill_val -seed $seed -prefix "PADI_"
    rm -rf $out_folder
    # python full_analysis.py -f $train_file -v $val_file -t $test_file -i -s MinMaxScaler -a $A -c 1 -k $K -ak $ak -o $out_folder -m $out_model -n  $model -hv $hill_val
    python make_summary_file.py -np $np -wp $wp -f $train_file -v $val_file -t $test_file -i -s MinMaxScaler -a $A -c 1 -k $K -ak $ak -m $out_model -n  $model -hv $hill_val -rf $results_file

done;
done;
done;

